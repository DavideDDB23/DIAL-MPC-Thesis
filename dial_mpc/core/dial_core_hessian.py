import os
import time
from dataclasses import dataclass
import importlib
import sys

import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
import art
import emoji

import jax
from jax import numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import functools

from brax.io import html
import brax.envs as brax_envs

import dial_mpc.envs as dial_envs
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.examples import examples
from dial_mpc.core.dial_config import DialConfig

# ----------------------------------------------------------------------------
# Debugging flag (set environment variable DIAL_DEBUG=1 to enable prints)
# ----------------------------------------------------------------------------
DEBUG = bool(int(os.environ.get("DIAL_DEBUG", "0")))


plt.style.use("science")
plt.rcParams['text.usetex'] = False

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

def cubic_newton_direction(g, H, sigma, tol: float = 1e-6, maxiter: int = 20):
    """Closed-form cubic-regularised Newton step.

    Solves   min_p  gᵀp + ½ pᵀHp + σ/6 ||p||³  using the scalar-\lambda root
    formulation (Nesterov & Polyak, 2006).  The returned vector is **minus**
    the minimiser so that existing call-sites that perform

        Y ← Y − direction

    continue to move along the Newton step.

    Parameters
    ----------
    g : (d,) jax.Array
        Gradient at current iterate.
    H : (d,d) jax.Array (symmetric)
        Hessian at current iterate.
    sigma : float
        Cubic regularisation parameter (>0).
    tol : float, optional
        Absolute tolerance on the λ-root.  Default 1e-6.
    maxiter : int, optional
        Maximum Newton iterations.  Default 50.

    Returns
    -------
    (d,) jax.Array
        Descent direction (−p★).
    """
    # Sanitize inputs to prevent upstream NaNs from propagating.
    g = jnp.nan_to_num(g)
    H = jnp.nan_to_num(H)

    # Force symmetry for numerical stability
    H = 0.5 * (H + H.T)

    eigvals, Q = jnp.linalg.eigh(H)
    g_hat = Q.T @ g

    # λ must be > -min(eigvals) to keep H+λI positive-definite.
    # Add a small cushion for stability.
    lambda_min = -jnp.min(eigvals)
    lambda0 = jnp.maximum(lambda_min, 0.0) + 1e-4

    def _newton_body(state):
        i, lmbda = state

        # p_hat in eigenbasis
        p_hat = -g_hat / (eigvals + lmbda)
        p_norm = jnp.linalg.norm(p_hat)

        # φ(λ), the root of which we are searching. Add eps for stability.
        phi = 1.0 / (p_norm + 1e-8) - sigma / (2.0 * (lmbda + 1e-8))

        # φ′(λ) derivative.
        p_hat_sq = p_hat**2
        denominators = eigvals + lmbda
        term = jnp.sum(jnp.nan_to_num(p_hat_sq / (denominators + 1e-8)))

        # Corrected derivative. The first term, d/dλ (1/||p||), is positive.
        # The previous implementation had a sign error here.
        dphi = term / (p_norm**3 + 1e-8) + sigma / (2.0 * (lmbda**2) + 1e-8)

        # Newton step. Add epsilon to dphi to avoid division by zero.
        # Since dphi should now be positive, a simple epsilon is sufficient.
        dphi_safe = dphi + 1e-8
        lmbda_next = lmbda - phi / dphi_safe

        # Keep λ in the valid region.
        return i + 1, jnp.maximum(lambda_min + 1e-6, lmbda_next)

    def _newton_cond(state):
        i, lmbda = state
        p_hat = -g_hat / (eigvals + lmbda)
        p_norm = jnp.linalg.norm(p_hat)
        phi = 1.0 / (p_norm + 1e-8) - sigma / (2.0 * (lmbda + 1e-8))
        return (jnp.abs(phi) > tol) & (i < maxiter)

    # Run Newton iterations with a JIT-able while_loop.
    i, lmbda_final = jax.lax.while_loop(_newton_cond, _newton_body, (0, lambda0))

    # Reconstruct final solution in original basis
    p_star = Q @ (-g_hat / (eigvals + lmbda_final))

    if DEBUG:
        jax.debug.print(
            "[solve_subproblem] iters={i}, lambda={l:.3e}, grad_norm={g:.3e}, p_norm={p:.3e}",
            i=i, l=lmbda_final, g=jnp.linalg.norm(g), p=jnp.linalg.norm(p_star)
        )

    return -jnp.nan_to_num(p_star)  # Return -p★ and sanitize output


def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return rews, pipline_states


@jax.jit
def softmax_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


class MBDPI:
    def __init__(self, args: DialConfig, env):
        self.args = args
        self.env = env
        self.nu = env.action_size

        self.update_fn = {
            "mppi": softmax_update,
        }[args.update_method]

        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = jnp.log(sigma1 / sigma0) / args.Ndiffuse
        self.sigmas = A * jnp.exp(B * jnp.arange(args.Ndiffuse))
        self.sigma_control = (
            args.horizon_diffuse_factor ** jnp.arange(args.Hnode + 1)[::-1]
        )

        # node to u
        self.ctrl_dt = 0.02
        self.step_us = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1)
        self.step_nodes = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1)
        self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)

        # cubic Newton step solver
        self.cubic_newton_direction = jax.jit(cubic_newton_direction)

        # setup function
        self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step))
        self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))
        self.node2u_vmap = jax.jit(
            jax.vmap(self.node2u, in_axes=(1), out_axes=(1))
        )  # process (horizon, node)
        self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=(1), out_axes=(1)))
        self.node2u_vvmap = jax.jit(
            jax.vmap(self.node2u_vmap, in_axes=(0))
        )  # process (batch, horizon, node)
        self.u2node_vvmap = jax.jit(jax.vmap(self.u2node_vmap, in_axes=(0)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=2)
        us = spline(self.step_us)
        return us

    @functools.partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        spline = InterpolatedUnivariateSpline(self.step_us, us, k=2)
        nodes = spline(self.step_nodes)
        return nodes

    @functools.partial(jax.jit, static_argnums=(0,))
    def reverse_once(self, state, rng, Ybar_i, noise_scale):

        '''
        Ybar_i: Current knots (N_nodes x N_control)
        State: Brax state
        rng: Random number generator
        noise_scale: sigma for isotropic noises (N_nodes)
        '''
        # jax.debug.print("Ybar_i:{}, noise_scale:{}", Ybar_i.shape, noise_scale.shape, ordered=True)
        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_Y = jax.random.normal(
            Y0s_rng, (self.args.Nsample, self.args.Hnode + 1, self.nu)
        )
        
        Y0s = eps_Y * noise_scale[None, :, None] 
   
        Y_ctrls = Y0s + Ybar_i
        Y_ctrls = jnp.clip(Y_ctrls, -1.0, 1.0)

        # we can't change the first control
        Y_ctrls = Y_ctrls.at[:, 0].set(Ybar_i[0, :])

        # Transform back the clipped eps_Y
        Y0s = Y_ctrls - Ybar_i
        eps_Y = Y0s / noise_scale[None, :, None]

        # convert Y_ctrls to us and add Ybar_i for baseline reward calculation
        Y_ctrls_with_base = jnp.concatenate([Y_ctrls, Ybar_i[None]], axis=0)
        us = self.node2u_vvmap(Y_ctrls_with_base)

        # esitimate mu_0tm1
        rewss, pipeline_statess = self.rollout_us_vmap(state, us)

        Y0s = jnp.reshape(Y0s, (self.args.Nsample, (self.args.Hnode+1)*self.nu)) # N_sample x (N_nodes*Nu)

        eps_Y = jnp.reshape(eps_Y, (self.args.Nsample, (self.args.Hnode+1)*self.nu))  # N_sample x (N_nodes*Nu)

        rew_Ybar_i = rewss[-1].mean()
        # Separate pipeline states for perturbed trajectories
        pipeline_statess_perturbed = jax.tree_util.tree_map(lambda x: x[:-1], pipeline_statess)
        qss = pipeline_statess_perturbed.q
        qdss = pipeline_statess_perturbed.qd
        xss = pipeline_statess_perturbed.x.pos
        rews = rewss[:-1].mean(axis=-1)

        # logp0 = (rews) / self.args.temp_sample
        logp0 = (rews - rew_Ybar_i) / (rews.std(axis=-1) * self.args.temp_sample)

        weights = jax.nn.softmax(logp0)
        # Ybar, new_noise_scale = self.update_fn(weights, Y0s, noise_scale, Ybar_i)

        gradient = (noise_scale[0] ** -1) * jnp.einsum("n,nk->k", weights, eps_Y)
        hessian = (noise_scale[0] ** -2) * (
            jnp.einsum("n,nk,nl->kl", weights, eps_Y, eps_Y)
            - jnp.einsum("n,nk->k", weights, eps_Y)[:, None]
            * jnp.einsum("n,nl->l", weights, eps_Y)[None, :]
        )

        hessian = -(self.args.temp_sample) * hessian
        gradient = -(self.args.temp_sample) * gradient
        # Hessian based 
        hessian = 0.5 * (hessian + jnp.transpose(hessian))
        
        eigenValues, U = jnp.linalg.eigh(hessian)
        
        # jax.debug.print("Eigenvalues: {}", eigenValues)

        eigenValues = jnp.clip(eigenValues, 1e-6, 100)

        hessian = U @ jnp.diag(eigenValues) @ jnp.transpose(U)

        direction = self.cubic_newton_direction(gradient, hessian, sigma=self.args.cubic_sigma)
        direction = jnp.reshape(direction, (self.args.Hnode + 1, self.nu))

        # NOTE: update only with reward
        alpha  = self.args.step_size_alpha
        Ybar = jnp.clip(Ybar_i - alpha * direction, -1.0, 1.0)

        qbar = jnp.einsum("n,nij->ij", weights, qss)
        qdbar = jnp.einsum("n,nij->ij", weights, qdss)
        xbar = jnp.einsum("n,nijk->ijk", weights, xss)

        info = {
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
            "new_noise_scale": noise_scale,
        }

        if DEBUG:
            grad_norm = jnp.linalg.norm(gradient)
            dir_norm = jnp.linalg.norm(direction)
            eig_min = eigenValues.min()
            jax.debug.print(
                "[reverse_once] grad_norm={:.3e}, dir_norm={:.3e}, eig_min={:.3e}, rew_mean={:.3e}",
                grad_norm,
                dir_norm,
                eig_min,
                rews.mean(),
            )

        return rng, Ybar, info

    def reverse(self, state, YN, rng):
        Yi = YN
        with tqdm(range(self.args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                t0 = time.time()
                rng, Yi, rews = self.reverse_once(
                    state, rng, Yi, self.sigmas[i] * jnp.ones(self.args.Hnode + 1)
                )
                Yi.block_until_ready()
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({"rew": f"{rews.mean():.2e}", "freq": f"{freq:.2f}"})
        return Yi

    @functools.partial(jax.jit, static_argnums=(0,))
    def shift(self, Y):
        u = self.node2u_vmap(Y)
        u = jnp.roll(u, -1, axis=0)
        u = u.at[-1].set(jnp.zeros(self.nu))
        Y = self.u2node_vmap(u)
        return Y

    def shift_Y_from_u(self, u, n_step):
        u = jnp.roll(u, -n_step, axis=0)
        u = u.at[-n_step:].set(jnp.zeros_like(u[-n_step:]))
        Y = self.u2node_vmap(u)
        return Y


def main():

    def reverse_scan(rng_Y0_state, factor):
        rng, Y0, state = rng_Y0_state
        rng, Y0, info = mbdpi.reverse_once(state, rng, Y0, factor)
        return (rng, Y0, state), info

    art.tprint("LeCAR @ CMU\nDIAL-MPC", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    config_or_example = parser.add_mutually_exclusive_group(required=True)
    config_or_example.add_argument("--config", type=str, default=None)
    config_or_example.add_argument("--example", type=str, default=None)
    config_or_example.add_argument("--list-examples", action="store_true")
    parser.add_argument(
        "--custom-env",
        type=str,
        default=None,
        help="Custom environment to import dynamically",
    )
    args = parser.parse_args()

    if args.list_examples:
        print("Examples:")
        for example in examples:
            print(f"  {example}")
        return

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.example is not None:
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(args.config))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    rng = jax.random.PRNGKey(seed=dial_config.seed)

    # find env config
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )

    print(emoji.emojize(":rocket:") + "Creating environment")
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    mbdpi = MBDPI(dial_config, env)

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    YN = jnp.zeros([dial_config.Hnode + 1, mbdpi.nu])

    rng_exp, rng = jax.random.split(rng)
    # Y0 = mbdpi.reverse(state_init, YN, rng_exp)
    Y0 = YN

    Nstep = dial_config.n_steps
    rews = []
    rews_plan = []
    rollout = []
    state = state_init
    us = []
    infos = []
    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            # forward single step
            state = step_env(state, Y0[0])
            rollout.append(state.pipeline_state)
            rews.append(state.reward)
            us.append(Y0[0])

            # update Y0
            Y0 = mbdpi.shift(Y0)

            n_diffuse = dial_config.Ndiffuse
            if t == 0:
                n_diffuse = dial_config.Ndiffuse_init
                print("Performing JIT on DIAL-MPC")

            t0 = time.time()
            traj_diffuse_factors = (
                mbdpi.sigma_control
                * dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
            )

            (rng, Y0, _), info = jax.lax.scan(
                reverse_scan, (rng, Y0, state), traj_diffuse_factors
            )
            rews_plan.append(info["rews"][-1].mean())
            infos.append(info)
            freq = 1 / (time.time() - t0)
            pbar.set_postfix({"rew": f"{state.reward:.2e}", "freq": f"{freq:.2f}"})

    rew = jnp.array(rews).mean()
    print(f"mean reward = {rew:.2e}")

    # save us
    # us = jnp.array(us)
    # jnp.save("./results/us.npy", us)

    # create result dir if not exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # plot rews_plan
    # plt.plot(rews_plan)
    # plt.savefig(os.path.join(dial_config.output_dir,
    #            f"{timestamp}_rews_plan.pdf"))

    # host webpage with flask
    print("Processing rollout for visualization")
    import flask

    app = flask.Flask(__name__)
    webpage = html.render(
        env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 1080, True
    )

    # save the html file
    with open(
        os.path.join(dial_config.output_dir, f"{timestamp}_brax_visualization.html"),
        "w",
    ) as f:
        f.write(webpage)

    # save the rollout
    data = []
    xdata = []
    for i in range(len(rollout)):
        pipeline_state = rollout[i]
        data.append(
            jnp.concatenate(
                [
                    jnp.array([i]),
                    pipeline_state.qpos,
                    pipeline_state.qvel,
                    pipeline_state.ctrl,
                ]
            )
        )
        xdata.append(infos[i]["xbar"][-1])
    data = jnp.array(data)
    xdata = jnp.array(xdata)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_states"), data)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_predictions"), xdata)

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)


if __name__ == "__main__":
    main()

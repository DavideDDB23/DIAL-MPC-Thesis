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
import mujoco

import numpy as np

import jax
from jax import numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import functools

from brax.io import html
from brax import math
import brax.envs as brax_envs
from dial_mpc.utils.function_utils import global_to_body_velocity

import dial_mpc.envs as dial_envs
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.examples import examples
from dial_mpc.core.dial_config import DialConfig
import scienceplots

plt.style.use(['science', 'no-latex'])

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


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

# JAX Butterworth biquad utilities (2nd-order)

def _butterworth_biquad_coeffs(fs: float, fc: float):
    ff = fc / fs
    ita = 1.0 / jnp.tan(jnp.pi * ff)
    q = jnp.sqrt(2.0)
    b0 = 1.0 / (1.0 + q * ita + ita ** 2)
    b1 = 2.0 * b0
    b2 = b0
    a1 = 2.0 * (ita ** 2 - 1.0) * b0
    a2 = -(1.0 - q * ita + ita ** 2) * b0
    b = jnp.array([b0, b1, b2])
    a = jnp.array([1.0, a1, a2])
    return b, a

@jax.jit
def _lfilter_df2t_biquad_2d(x2d: jax.Array, b: jax.Array, a: jax.Array):
    b0, b1, b2 = b
    _, a1, a2 = a

    def step(carry, x_t):
        z1, z2 = carry
        y_t = b0 * x_t + z1
        z1_new = b1 * x_t + z2 - a1 * y_t
        z2_new = b2 * x_t - a2 * y_t
        return (z1_new, z2_new), y_t

    T, C = x2d.shape
    carry0 = (jnp.zeros((C,), x2d.dtype), jnp.zeros((C,), x2d.dtype))
    (_, _), y2d = jax.lax.scan(step, carry0, x2d)
    return y2d


def _filtfilt_biquad(signal: jax.Array, b: jax.Array, a: jax.Array, axis: int):
    x = jnp.moveaxis(signal, axis, 0)
    T = x.shape[0]
    batch_shape = x.shape[1:]
    C = int(np.prod(batch_shape))
    x2d = x.reshape(T, C)

    padlen = int(min(3 * (b.shape[0] - 1), T - 1))
    if padlen <= 0:
        y2d = _lfilter_df2t_biquad_2d(x2d, b, a)
        y = y2d.reshape((T,) + batch_shape)
        return jnp.moveaxis(y, 0, axis)

    left = 2 * x2d[0:1] - x2d[1 : padlen + 1][::-1]
    right = 2 * x2d[-1:] - x2d[-padlen - 1 : -1][::-1]
    xpad = jnp.concatenate([left, x2d, right], axis=0)

    y_fwd = _lfilter_df2t_biquad_2d(xpad, b, a)
    y_bwd = _lfilter_df2t_biquad_2d(y_fwd[::-1], b, a)[::-1]
    y2d = y_bwd[padlen : padlen + T]
    y = y2d.reshape((T,) + batch_shape)
    return jnp.moveaxis(y, 0, axis)


def second_order_butterworth(
    signal: jax.Array,
    f_sampling: float = 100.0,
    f_cutoff: float = 15.0,
    method: str = "forward_backward",
    axis: int = -1,
) -> jax.Array:
    b, a = _butterworth_biquad_coeffs(f_sampling, f_cutoff)
    b = b.astype(signal.dtype)
    a = a.astype(signal.dtype)

    if method == "forward_backward":
        return _filtfilt_biquad(signal, b, a, axis=axis)
    elif method == "forward":
        x = jnp.moveaxis(signal, axis, 0)
        T = x.shape[0]
        batch_shape = x.shape[1:]
        C = int(np.prod(batch_shape))
        y2d = _lfilter_df2t_biquad_2d(x.reshape(T, C), b, a)
        y = y2d.reshape((T,) + batch_shape)
        return jnp.moveaxis(y, 0, axis)
    elif method == "backward":
        return jnp.flip(
            second_order_butterworth(
                jnp.flip(signal, axis=axis),
                f_sampling,
                f_cutoff,
                method="forward",
                axis=axis,
            ),
            axis=axis,
        )
    else:
        raise ValueError("method must be 'forward', 'backward', or 'forward_backward'")


def jax_colored_noise(key, beta: float, shape: tuple, axis: int = -1) -> jax.Array:
    """
    Generate colored noise with power spectrum ~ 1/f^beta using JAX.
    Ensures zero DC (mean) and unit variance along the specified axis.
    """
    T = shape[axis]

    # Frequencies for rFFT
    freqs = jnp.fft.rfftfreq(T)

    # Amplitude spectrum ~ 1/f^(beta/2), with zero DC
    amp = jnp.where(freqs == 0.0, 0.0, freqs ** (-beta / 2.0))

    # Generate white noise and FFT
    white = jax.random.normal(key, shape)
    white_fft = jnp.fft.rfft(white, axis=axis)

    # Reshape amplitude for broadcasting across all non-time axes
    amp_shape = [1] * len(shape)
    amp_shape[axis] = white_fft.shape[axis]
    amp = amp.reshape(amp_shape)

    # Apply spectral shaping and invert
    colored_fft = white_fft * amp
    colored = jnp.fft.irfft(colored_fft, n=T, axis=axis)

    # Remove any residual mean and normalize to unit variance along time axis
    colored = colored - jnp.mean(colored, axis=axis, keepdims=True)
    colored = colored / (jnp.std(colored, axis=axis, keepdims=True) + 1e-8)

    return colored


def jax_bandlimited_noise(key, fmax: float, dt: float, shape: tuple, axis: int = 1, num_harmonics: int = 8) -> jax.Array:
    """
    Generate band-limited noise by summing a small number of random Fourier modes up to fmax.
    Expected shape is (batch, T, D) with time at axis=1 by default.
    """
    assert axis == 1, "jax_bandlimited_noise expects time axis at position 1"
    batch, T, D = shape
    two_pi = 2.0 * jnp.pi

    key_f, key_phi, key_amp = jax.random.split(key, 3)
    # Sample frequencies uniformly in (0, fmax]
    freqs = jax.random.uniform(key_f, (batch, num_harmonics), minval=1e-3, maxval=fmax)
    phases = two_pi * jax.random.uniform(key_phi, (batch, num_harmonics, 1, 1))
    amps = jax.random.normal(key_amp, (batch, num_harmonics, 1, D)) / jnp.sqrt(num_harmonics)

    t = (jnp.arange(T) * dt)[None, None, :, None]
    omega_t = two_pi * freqs[:, :, None, None] * t
    waves = jnp.cos(omega_t + phases)
    series = jnp.sum(amps * waves, axis=1)  # (batch, T, D)

    # Zero-mean and unit-variance along time axis for each (batch, D)
    series = series - jnp.mean(series, axis=1, keepdims=True)
    series = series / (jnp.std(series, axis=1, keepdims=True) + 1e-8)
    return series


class MBDPI:
    def __init__(self, cli_args, args: DialConfig, env, rng):
        self.cli_args = cli_args
        self.args = args
        self.env = env
        self.nu = env.action_size
        self.rng = rng

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
        self.u2node_vvmap = jax.jit(jax.vmap(self.node2u_vmap, in_axes=(0)))

        # VIGAS or DIAL specific initialization
        self.optimizer = cli_args.optimizer
        if self.optimizer == "vigas":
            # --- VIGAS (Variational Inference Guided Annealing Search) Initialization ---
            # VIGAS replaces isotropic sampling with an adaptive low-rank Gaussian distribution.
            # This allows the optimizer to learn the shape of the reward landscape online.
            # The variational distribution is q(Y) = N(mu, L @ L.T + diag(d)).
            self.k = cli_args.vigas_rank  # Rank of the covariance matrix
            D = (args.Hnode + 1) * self.nu  # Dimension of the flattened trajectory

            # Initialize mean trajectory (mu)
            mu_init = jnp.ones([args.Hnode + 1, self.nu]) * env.default_action
            # Initialize low-rank covariance factor (L)
            L_init = jnp.zeros((D, self.k))
            # Initialize diagonal covariance factor (d)
            d_init = jnp.ones(D) * cli_args.vigas_init_var
            self.q_params_init = (mu_init, L_init, d_init)
        else:
            # --- DIAL-MPC Initialization ---
            self.noise_type = cli_args.noise_type if cli_args.noise_type is not None else args.noise_type
            
            noise_shape = (100 * self.args.Nsample, self.args.Hnode + 1, self.nu)

            if self.noise_type == "colored":
                noise_beta = self.cli_args.beta
                if noise_beta is None:
                    noise_beta = 1.0
                rng, noise_rng = jax.random.split(self.rng)
                self.rng = rng
                self.noise = jax_colored_noise(noise_rng, noise_beta, noise_shape, axis=1)
                self.normal_noise = None 
                self.lp_noise = None
            elif self.noise_type == "bandlimited":
                cutoff_freq = self.cli_args.lpfreq
                if cutoff_freq is None:
                    cutoff_freq = 2.0
                rng, noise_rng = jax.random.split(self.rng)
                self.rng = rng
                self.noise = jax_bandlimited_noise(
                    noise_rng,
                    fmax=float(cutoff_freq),
                    dt=float(self.node_dt),
                    shape=noise_shape,
                    axis=1,
                    num_harmonics=8,
                )
                self.normal_noise = None
                self.lp_noise = None
            else:
                rng, noise_rng = jax.random.split(self.rng)
                self.rng = rng
                normal_noise = jax.random.normal(noise_rng, noise_shape)
                self.normal_noise = normal_noise
                self.noise = self.normal_noise

                # low pass filter
                if self.noise_type == "lp":
                    cutoff_freq = self.cli_args.lpfreq
                    if cutoff_freq is None:
                        cutoff_freq = 2.0
                    sampling_freq = 1.0 / self.node_dt
                    lp_noise = second_order_butterworth(
                        self.normal_noise,
                        f_sampling=float(sampling_freq),
                        f_cutoff=float(cutoff_freq),
                        method="forward_backward",
                        axis=-2,
                    )
                    scale = lp_noise.std(axis=0, keepdims=True)
                    lp_noise_normalized = lp_noise / (scale + 1e-8)
                    self.lp_noise = lp_noise_normalized
                    self.noise = self.lp_noise

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
    def vigas_update_fn(self, state, rng, q_params, temp):
        # This function performs one step of VIGAS.
        # It samples from the current variational distribution, evaluates the samples,
        # and then updates the distribution's parameters using a natural gradient step.
        mu, L, d = q_params
        rng_k, rng_d, rng_update = jax.random.split(rng, 3)
        D = (self.args.Hnode + 1) * self.nu

        # 1. Sample from q(Y; mu, L, d) = N(mu, L @ L.T + diag(d))
        # This creates structured, correlated noise that adapts to the problem.
        mu_flat = mu.flatten()
        eps_k = jax.random.normal(rng_k, (self.args.Nsample, self.k))
        eps_d = jax.random.normal(rng_d, (self.args.Nsample, D))
        # Low-rank sampling: Y = mu + L*eps_k + diag(sqrt(d))*eps_d
        Y_samples_flat = mu_flat[None, :] + eps_k @ L.T + eps_d * jnp.sqrt(d)[None, :]
        Y0s = Y_samples_flat.reshape(self.args.Nsample, self.args.Hnode + 1, self.nu)

        # As in DIAL, fix the first control of all samples to be that of the current mean.
        # This grounds the optimization at the current state, enforcing MPC principles.
        Y0s = Y0s.at[:, 0, :].set(mu[0, :])
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # 2. Evaluate samples
        us = self.node2u_vvmap(Y0s)
        rewss, pipeline_statess = self.rollout_us_vmap(state, us)
        rews = rewss.mean(axis=-1)

        # 3. Update Variational Parameters via Natural Gradient
        # This step maximizes the Evidence Lower Bound (ELBO) by fitting the distribution
        # to the high-reward samples. The weights are from the softmax of rewards,
        # which makes this a natural gradient update.
        scores = jax.lax.cond(
            self.cli_args.vigas_standardize,
            lambda r: (r - jnp.mean(r)) / (jnp.std(r) + 1e-6),
            lambda r: r,
            rews,
        )
        weights = jax.nn.softmax(scores / temp)


        # Update Mean: a weighted average of the samples.
        mu_new_raw = jnp.einsum("n,nij->ij", weights, Y0s)
        mu_new_flat_raw = mu_new_raw.flatten()

        # Update Covariance (L and d)
        # Center samples around the new mean to compute the sample covariance.
        centered_Y_flat = Y_samples_flat - mu_new_flat_raw[None, :]

        # Get top-k eigenvectors of the weighted sample covariance matrix.
        # This is done efficiently via SVD of the sqrt-weighted centered samples matrix.
        weighted_centered_Y = centered_Y_flat * jnp.sqrt(weights)[:, None]
        _, S, Vt = jnp.linalg.svd(weighted_centered_Y, full_matrices=False)

        # The new low-rank factor L is the scaled principal components.
        L_new_raw = Vt[:self.k, :].T * S[:self.k][None, :]

        # The new diagonal factor d is the residual variance.
        # It's the weighted sample variance minus variance explained by the low-rank part.
        total_sample_variance = jnp.einsum('n,ni->i', weights, centered_Y_flat**2)
        variance_from_L = jnp.sum(L_new_raw**2, axis=1)
        d_new_raw = jnp.maximum(total_sample_variance - variance_from_L, self.cli_args.vigas_cov_floor)

        # Apply EMA-based trust region to stabilize updates
        mu_new = (1 - self.cli_args.vigas_mu_alpha) * mu + self.cli_args.vigas_mu_alpha * mu_new_raw
        L_new = (1 - self.cli_args.vigas_cov_alpha) * L + self.cli_args.vigas_cov_alpha * L_new_raw
        d_new = (1 - self.cli_args.vigas_cov_alpha) * d + self.cli_args.vigas_cov_alpha * d_new_raw

        q_params_new = (mu_new, L_new, d_new)

        # Also return info for logging, similar to reverse_once
        qbar = jnp.einsum("n,nij->ij", weights, pipeline_statess.q)
        qdbar = jnp.einsum("n,nij->ij", weights, pipeline_statess.qd)
        xbar = jnp.einsum("n,nijk->ijk", weights, pipeline_statess.x.pos)

        info = {
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
            "new_noise_scale": d_new.mean(), # For compatibility with info structure
        }
        return rng_update, q_params_new, info


    @functools.partial(jax.jit, static_argnums=(0,))
    def reverse_once(self, state, rng, Ybar_i, noise_scale):
        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        #eps_Y = jax.random.normal(
        #    Y0s_rng, (self.args.Nsample, self.args.Hnode + 1, self.nu)
        #)
        eps_Y = jax.random.choice(Y0s_rng, self.noise, shape=(self.args.Nsample,), axis=0)
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        # we can't change the first control
        Y0s = Y0s.at[:, 0].set(Ybar_i[0, :])
        # append Y0s with Ybar_i to also evaluate Ybar_i
        Y0s = jnp.concatenate([Y0s, Ybar_i[None]], axis=0)
        Y0s = jnp.clip(Y0s, -1.0, 1.0)
        # convert Y0s to us
        us = self.node2u_vvmap(Y0s)

        # esitimate mu_0tm1
        rewss, pipeline_statess = self.rollout_us_vmap(state, us)
        rew_Ybar_i = rewss[-1].mean()
        qss = pipeline_statess.q
        qdss = pipeline_statess.qd
        xss = pipeline_statess.x.pos
        rews = rewss.mean(axis=-1)
        logp0 = (rews - rew_Ybar_i) / rews.std(axis=-1) / self.args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar, new_noise_scale = self.update_fn(weights, Y0s, noise_scale, Ybar_i)

        # NOTE: update only with reward
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)
        qbar = jnp.einsum("n,nij->ij", weights, qss)
        qdbar = jnp.einsum("n,nij->ij", weights, qdss)
        xbar = jnp.einsum("n,nijk->ijk", weights, xss)

        info = {
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
            "new_noise_scale": new_noise_scale,
        }

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

    @functools.partial(jax.jit, static_argnums=(0,))
    def shift_q_params(self, q_params):
        # This function shifts the variational parameters for the receding horizon.
        mu, L, d = q_params
        # Shift mean trajectory using the existing spline-based shift
        mu_shifted = self.shift(mu)

        # Shift covariance factors. This is done by reshaping to trajectory shape,
        # rolling along the time axis, and flattening back.
        D_traj_shape = (self.args.Hnode + 1, self.nu)

        # Shift L (low-rank factor)
        L_reshaped = L.reshape(D_traj_shape + (self.k,))
        L_shifted_reshaped = jnp.roll(L_reshaped, -1, axis=0)
        L_shifted_reshaped = L_shifted_reshaped.at[-1, :, :].set(jnp.zeros((self.nu, self.k)))
        L_shifted = L_shifted_reshaped.reshape((-1, self.k))

        # Shift d (diagonal factor) and re-initialize the variance for the new last step.
        d_reshaped = d.reshape(D_traj_shape)
        d_shifted_reshaped = jnp.roll(d_reshaped, -1, axis=0)
        d_shifted_reshaped = d_shifted_reshaped.at[-1, :].set(jnp.ones(self.nu) * self.cli_args.vigas_init_var)
        d_shifted = d_shifted_reshaped.flatten()

        return (mu_shifted, L_shifted, d_shifted)

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

    def vigas_scan(rng_q_params_state, temp):
        rng, q_params, state = rng_q_params_state
        rng, q_params, info = mbdpi.vigas_update_fn(state, rng, q_params, temp)
        return (rng, q_params, state), info

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
    parser.add_argument(
        "--port", type=int, default=5000, help="Port number for visualization"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed"
    )
    parser.add_argument(
        "--lpfreq", type=float, default=None, help="Low pass frequency"
    )
    parser.add_argument(
        "--beta", type=float, default=None, help="Colored noise parameter"
    )
    parser.add_argument(
        "--noise-type", type=str, default=None, help="Type of noise to use for DIAL (lp, colored, bandlimited, or none)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="dial", help="Optimizer to use: 'dial' or 'vigas'"
    )
    
    # --- Arguments for VIGAS Optimizer ---
    parser.add_argument(
        "--vigas_rank", type=int, default=10, help="[VIGAS] Rank of the covariance matrix"
    )
    parser.add_argument(
        "--vigas_init_var", type=float, default=0.1, help="[VIGAS] Initial variance for the diagonal covariance"
    )
    parser.add_argument(
        "--vigas_temp", type=float, default=0.01, help="[VIGAS] Temperature for softmax weighting"
    )
    parser.add_argument(
        "--vigas_standardize", action="store_true", help="[VIGAS] Standardize rewards before softmax"
    )
    parser.add_argument(
        "--vigas_mu_alpha", type=float, default=1.0, help="[VIGAS] EMA factor for mean update (trust region)"
    )
    parser.add_argument(
        "--vigas_cov_alpha", type=float, default=1.0, help="[VIGAS] EMA factor for covariance update (trust region)"
    )
    parser.add_argument(
        "--vigas_cov_floor", type=float, default=1e-6, help="[VIGAS] Floor for diagonal covariance"
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
    rng = jax.random.PRNGKey(seed=args.seed if args.seed is not None else dial_config.seed)

    # find env config
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )

    print(emoji.emojize(":rocket:") + "Creating environment")
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    rng, mbdpi_rng = jax.random.split(rng)
    mbdpi = MBDPI(args, dial_config, env, mbdpi_rng)

    optimizer_display = "VIGAS" if args.optimizer == "vigas" else f"DIAL-MPC with {mbdpi.noise_type} noise"
    yaml_file_name = args.example if args.example is not None else args.config
    print(f"Running with {optimizer_display} on {yaml_file_name}")

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)
    # Print crate height at start (for crate climb scenario)
    if dial_config.env_name == "unitree_go2_crate_climb":
        try:
            model = env.sys.mj_model
            data = mujoco.MjData(model)
            # Ensure mocap body's pose is set from model before forward kinematics
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "box_body")
            mocap_id = model.body_mocapid[body_id]
            if mocap_id != -1:
                data.mocap_pos[mocap_id] = model.body_pos[body_id]
                data.mocap_quat[mocap_id] = model.body_quat[body_id]
            mujoco.mj_forward(model, data)
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, "static_box")
            geom_center = data.geom_xpos[geom_id]
            R = np.array(data.geom_xmat[geom_id]).reshape(3, 3)
            half_sizes = np.array(model.geom_size[geom_id])
            # Robust world Z top (handles rotations): center_z + sum(|R[2,:]| * half_sizes)
            crate_top_z = float(geom_center[2] + float(np.sum(np.abs(R[2, :]) * half_sizes)))
            print(f"Crate height (top above ground) = {crate_top_z:.3f} m")
        except Exception as e:
            print(f"Could not compute crate height: {e}")
    # Use the environment torso index (as in the env) for body-frame velocity/yaw
    torso_idx = getattr(env, "_torso_idx", 1) - 1
    # Compute walk-tracking error only for the unitree_go2_trot example
    compute_walk_tracking = (args.example == "unitree_go2_trot")
    # Compute sequential jumping metric for unitree_go2_seq_jump tasks
    compute_seq_jumping = (dial_config.env_name == "unitree_go2_seq_jump")

    # Initialize optimizer state
    if args.optimizer == 'vigas':
        q_params = mbdpi.q_params_init
        Y0 = q_params[0] # For initial step and logging
    else:
        Y0 = jnp.ones([dial_config.Hnode + 1, mbdpi.nu]) * env.default_action

    Nstep = dial_config.n_steps
    rews = []
    rews_plan = []
    rollout = []
    state = state_init
    us = []
    infos = []
    freqs = []
    tracking_errors = []
    seq_jumping_states = []
    vel_tars = []
    ang_vel_tars = []
    z_feet_tars = []
    # Collect per-step convergence traces: best reward per inner iteration
    convergence_traces = []  # list of 1D numpy arrays, length = n_diffuse for the step
    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            # forward single step (MPC): execute only the first control, then shift the horizon
            if args.optimizer == 'vigas':
                # VIGAS: use the first action from the mean of the variational distribution q (mu[0])
                state = step_env(state, q_params[0][0])
                rollout.append(state.pipeline_state)
                rews.append(state.reward)
                us.append(q_params[0][0])
                # Shift variational parameters (mu, L, d) to maintain receding-horizon consistency
                q_params = mbdpi.shift_q_params(q_params)
            else:
                # DIAL-MPC: use the first action from the current nominal denoised plan Y0
                state = step_env(state, Y0[0])
                rollout.append(state.pipeline_state)
                rews.append(state.reward)
                us.append(Y0[0])
                # Shift nominal trajectory forward (spline-consistent shift)
                Y0 = mbdpi.shift(Y0)

            # record target commands for analysis
            vel_tars.append(state.info["vel_tar"])
            ang_vel_tars.append(state.info["ang_vel_tar"])
            # record target feet heights if available
            try:
                z_feet_tars.append(state.info["z_feet_tar"])  # shape (4,)
            except Exception:
                pass

            # compute per-step walk-tracking error (body-frame vx, vy, yaw-rate) only for unitree_go2_trot
            if compute_walk_tracking:
                x = state.pipeline_state.x
                xd = state.pipeline_state.xd
                yaw = math.quat_to_euler(x.rot[torso_idx])[2]
                vb = global_to_body_velocity(
                    xd.vel[torso_idx], x.rot[torso_idx]
                )
                ab = global_to_body_velocity(
                    xd.ang[torso_idx] * jnp.pi / 180.0, x.rot[torso_idx]
                )
                vel_tar = state.info["vel_tar"]
                ang_vel_tar = state.info["ang_vel_tar"]
                err = jnp.sqrt(
                    (vb[0] - vel_tar[0]) ** 2
                    + (vb[1] - vel_tar[1]) ** 2
                    + (ab[2] - ang_vel_tar[2]) ** 2
                )
                tracking_errors.append(err)

            # Collect data for sequential jumping metric calculation
            if compute_seq_jumping:
                # Store full state info needed for contact reward calculation
                seq_jumping_states.append({
                    'pipeline_state': state.pipeline_state,
                    'contact_stage': state.info["contact_stage"],
                    'contact_targets': state.info["contact_targets"],
                    'contact_target_radius': state.info["contact_target_radius"],
                    'step': state.info["step"]
                })


            n_diffuse = dial_config.Ndiffuse
            if t == 0:
                n_diffuse = dial_config.Ndiffuse_init
                print(f"Performing JIT on {optimizer_display}")

            t0 = time.time()
            if args.optimizer == 'vigas':
                # For VIGAS, we use a constant temperature across inner updates.
                # Exploration anneals implicitly as the learned covariance (L, d) sharpens
                # via the reward-weighted SVD fit (plus optional EMA trust region).
                temps = jnp.ones(n_diffuse) * args.vigas_temp
                (rng, q_params, _), info = jax.lax.scan(
                    vigas_scan, (rng, q_params, state), temps
                )
            else:
                # DIAL-MPC uses an explicit diffusion/denoising schedule:
                # per-step noise scales decay with traj_diffuse_factor and horizon shaping
                # (sigma_control), then reverse diffusion refines the nominal plan.
                traj_diffuse_factors = (
                    mbdpi.sigma_control * dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
                )
                (rng, Y0, _), info = jax.lax.scan(
                    reverse_scan, (rng, Y0, state), traj_diffuse_factors
                )

            rews_plan.append(info["rews"][-1].mean())
            # Convergence trace for this step: best reward per inner iteration, cumulative best
            try:
                # info["rews"] has shape (n_diffuse, Nsample[+1])
                inner_best = jnp.max(info["rews"], axis=-1)  # (n_diffuse,)
                inner_best_cum = jnp.maximum.accumulate(inner_best)
                convergence_traces.append(np.asarray(inner_best_cum))
            except Exception:
                pass
            infos.append(info)
            freq = 1 / (time.time() - t0)
            freqs.append(freq)
            pbar.set_postfix({"rew": f"{state.reward:.2e}", "freq": f"{freq:.2f}"})

    rew = jnp.array(rews).mean()
    print(f"mean reward {optimizer_display}= {rew:.4e}")
    freqs = jnp.array(freqs)
    print(f"mean freq {optimizer_display} = {freqs.mean():.2f}")
    print(f"median freq {optimizer_display} = {jnp.median(freqs):.2f}")

    us_arr = jnp.array(us)
    control_variation = jnp.sum(jnp.linalg.norm(jnp.diff(us_arr, axis=0), axis=1))
    print(f"Control variation {optimizer_display} = {control_variation:.2e}")

    if compute_walk_tracking and len(tracking_errors) > 0:
        tracking_errors = jnp.array(tracking_errors)
        walk_track_error = tracking_errors.mean()
        print(f"Walk-Tracking tracking error (lower is better) = {walk_track_error:.3f}")

    # Calculate sequential jumping metric if applicable  
    if compute_seq_jumping and len(seq_jumping_states) > 0:
        def calculate_contact_reward_paper_formula(state_data, wcorrect=0.1, wwrong=0.1):
            """Calculate contact reward according to paper formula:
            r(j)_con(t) = wcorrect * n(j)_correct(t) - wwrong * [n(j)_wrong(t) - n(j-1)_correct(t)]
            Using correct foot position detection like the base environment.
            """
            pipeline_state = state_data['pipeline_state']
            current_stage = int(state_data['contact_stage'])
            contact_targets = state_data['contact_targets']
            contact_target_radius = state_data['contact_target_radius']
            
            # Use the same contact detection as the existing UnitreeGo2SeqJumpEnv
            n_correct_current = 0
            n_wrong_current = 0
            n_correct_previous = 0
            
            # Count contacts for current stage j
            for foot_idx in range(4):
                contact_dist = pipeline_state.contact.dist[foot_idx]
                contact_pt = pipeline_state.contact.pos[foot_idx]
                
                # Check if foot is in contact
                is_in_contact = contact_dist <= 0.001
                
                if is_in_contact:
                    # Check if contact is within target radius for current stage j
                    if current_stage < len(contact_targets):
                        target_pos = contact_targets[current_stage, foot_idx, :2]  # x, y only
                        dist_to_target_sq = jnp.sum((contact_pt[:2] - target_pos) ** 2)
                        target_radius_sq = contact_target_radius[current_stage, foot_idx] ** 2
                        
                        if dist_to_target_sq <= target_radius_sq:
                            n_correct_current += 1
                        else:
                            n_wrong_current += 1
                    else:
                        n_wrong_current += 1  # Current stage doesn't exist
                        
                    # Check if contact would be correct for previous stage (j-1)
                    if current_stage > 0:
                        prev_target_pos = contact_targets[current_stage - 1, foot_idx, :2]
                        dist_to_prev_target_sq = jnp.sum((contact_pt[:2] - prev_target_pos) ** 2)
                        prev_target_radius_sq = contact_target_radius[current_stage - 1, foot_idx] ** 2
                        
                        if dist_to_prev_target_sq <= prev_target_radius_sq:
                            n_correct_previous += 1
            
            # Apply paper's formula: r(j)_con(t) = wcorrect * n(j)_correct(t) - wwrong * [n(j)_wrong(t) - n(j-1)_correct(t)]
            contact_reward = wcorrect * n_correct_current - wwrong * (n_wrong_current - n_correct_previous)
            return contact_reward
        
        # Calculate contact rewards for all timesteps
        contact_rewards = []
        for state_data in seq_jumping_states:
            reward = calculate_contact_reward_paper_formula(state_data)
            contact_rewards.append(reward)
        
        contact_rewards = jnp.array(contact_rewards)
        
        # Calculate per-stage minimum rewards according to paper
        # Only consider the stable contact period of each stage, not the jumping phase
        jump_dt = env._config.jump_dt if hasattr(env._config, 'jump_dt') else 0.8
        dt = env.dt
        timesteps_per_stage = int(jump_dt / dt)  # 0.8 / 0.02 = 40 timesteps per stage
        
        # Consider only the last portion of each stage when robot should be stable on target
        stable_window = int(0.3 / dt)  # 0.3 / 0.02 = 15 timesteps
        
        stage_min_rewards = []
        num_stages = 10 
        
        for stage in range(num_stages):
            stage_start = stage * timesteps_per_stage
            stage_end = min((stage + 1) * timesteps_per_stage, len(contact_rewards))
            
            # Only consider the stable window at the end of each stage
            stable_start = max(stage_start, stage_end - stable_window)
            stable_end = stage_end
            
            if stable_start < len(contact_rewards) and stable_start < stable_end:
                stable_rewards = contact_rewards[stable_start:stable_end]
                if len(stable_rewards) > 0:
                    stage_min_reward = float(jnp.min(stable_rewards))
                    stage_min_rewards.append(stage_min_reward)
                else:
                    stage_min_rewards.append(0.0)
            else:
                stage_min_rewards.append(0.0)
        
        # Calculate total contact reward (sum of stage minimums)
        total_contact_reward = sum(stage_min_rewards)
        
        # Normalize to [0,1] range as shown in paper
        # Maximum possible reward per stage: wcorrect * 4 feet = 0.1 * 4 = 0.4
        # Maximum possible total over 10 stages: 0.4 * 10 = 4.0  
        # But we need to map the range [-4.0, 4.0] to [0, 1]
        max_possible_per_stage = 0.1 * 4  # All feet correct
        min_possible_per_stage = -0.1 * 4  # All feet wrong
        max_possible_total = max_possible_per_stage * num_stages  # 4.0
        min_possible_total = min_possible_per_stage * num_stages  # -4.0
        
        # Normalize from [-4.0, 4.0] to [0, 1]
        normalized_total = (total_contact_reward - min_possible_total) / (max_possible_total - min_possible_total)
        normalized_total = max(0.0, min(1.0, normalized_total))  # Clamp to [0,1]
        
        print(f"Sequential Jumping - Raw Total Contact Reward = {total_contact_reward:.3f}")
        print(f"Sequential Jumping - Normalized Total Contact Reward [0,1] = {normalized_total:.3f}")

    # Evaluate success/failure for crate climb before visualization
    if dial_config.env_name == "unitree_go2_crate_climb":
        try:
            model = env.sys.mj_model
            data = mujoco.MjData(model)
            # Sync mocap pose for the crate body before forward
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "box_body")
            mocap_id = model.body_mocapid[body_id]
            if mocap_id != -1:
                data.mocap_pos[mocap_id] = model.body_pos[body_id]
                data.mocap_quat[mocap_id] = model.body_quat[body_id]
            mujoco.mj_forward(model, data)
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, "static_box")
            geom_center = data.geom_xpos[geom_id]
            R = np.array(data.geom_xmat[geom_id]).reshape(3, 3)
            half_sizes = np.array(model.geom_size[geom_id])

            crate_center_xy = np.array(geom_center[:2])
            crate_top_z = float(geom_center[2] + float(np.sum(np.abs(R[2, :]) * half_sizes)))
            # Projected half-size in world XY for conservative bounds
            half_sizes_xy = np.abs(R[:2, :]) @ half_sizes
            half_size_xy = half_sizes_xy
            margin = 0.1
            x_min = float(crate_center_xy[0] - half_size_xy[0] + margin)
            x_max = float(crate_center_xy[0] + half_size_xy[0] - margin)
            y_min = float(crate_center_xy[1] - half_size_xy[1] + margin)
            y_max = float(crate_center_xy[1] + half_size_xy[1] - margin)

            # Use the last 1.0s window to assess stability on top of the crate
            T_total = len(rollout)
            window_len = max(1, min(T_total, int(0.1 / float(env.dt))))
            start_idx = T_total - window_len
            base_pos = jnp.stack([rollout[t].x.pos[torso_idx] for t in range(start_idx, T_total)], axis=0)
            base_rot = jnp.stack([rollout[t].x.rot[torso_idx] for t in range(start_idx, T_total)], axis=0)
            base_vel = jnp.stack([rollout[t].xd.vel[torso_idx] for t in range(start_idx, T_total)], axis=0)
            rpy = jax.vmap(math.quat_to_euler)(base_rot)

            z_ok = base_pos[:, 2] > (crate_top_z + 0.20)
            xy_ok = (
                (base_pos[:, 0] > x_min)
                & (base_pos[:, 0] < x_max)
                & (base_pos[:, 1] > y_min)
                & (base_pos[:, 1] < y_max)
            )
            roll_ok = jnp.abs(rpy[:, 0]) < 0.35  # ~20 deg
            pitch_ok = jnp.abs(rpy[:, 1]) < 0.35  # ~20 deg
            speed_ok = jnp.linalg.norm(base_vel, axis=1) < 0.25
            stable = z_ok & xy_ok & roll_ok & pitch_ok & speed_ok
            success = bool(jnp.all(stable))
            result_str = "SUCCESS" if success else "FAILURE"
            print(f"{result_str}: crate height = {crate_top_z:.3f} m")
        except Exception as e:
            print(f"Could not evaluate crate success: {e}")

    # create result dir if not exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # plot rews_plan
    # plt.plot(rews_plan)
    # plt.savefig(os.path.join(dial_config.output_dir,
    #             f"{timestamp}_rews_plan.pdf"))

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

    # Save consolidated rollout metrics for analysis (e.g., behavioral trajectories)
    try:
        T_total = len(rollout)
        time_arr = jnp.arange(T_total) * float(env.dt)
        # Torso kinematics
        base_pos = jnp.stack([rollout[t].x.pos[torso_idx] for t in range(T_total)], axis=0)
        base_rot = jnp.stack([rollout[t].x.rot[torso_idx] for t in range(T_total)], axis=0)
        base_vel_world = jnp.stack([rollout[t].xd.vel[torso_idx] for t in range(T_total)], axis=0)
        rpy = jax.vmap(math.quat_to_euler)(base_rot)
        yaw_arr = rpy[:, 2]
        # Body-frame linear velocity (vx, vy, vz)
        base_vel_body = jax.vmap(global_to_body_velocity)(base_vel_world, base_rot)
        # Feet heights (z) for the 4 feet sites
        feet_site_ids = getattr(env, "_feet_site_id", None)
        feet_z = None
        if feet_site_ids is not None:
            feet_z = jnp.stack([rollout[t].site_xpos[feet_site_ids][:, 2] for t in range(T_total)], axis=0)
        # Joint angles (exclude free joint)
        joint_angles = jnp.stack([rollout[t].qpos[7:] for t in range(T_total)], axis=0)
        # Controls executed (already collected during rollout)
        us_arr = jnp.array(us)
        # Targets recorded during rollout
        vel_tar_series = jnp.array(vel_tars) if len(vel_tars) == T_total else jnp.zeros((T_total, 3))
        ang_vel_tar_series = jnp.array(ang_vel_tars) if len(ang_vel_tars) == T_total else jnp.zeros((T_total, 3))

        np.savez(
            os.path.join(dial_config.output_dir, f"{timestamp}_rollout_metrics.npz"),
            time=np.asarray(time_arr),
            base_pos=np.asarray(base_pos),
            base_vel_world=np.asarray(base_vel_world),
            base_vel_body=np.asarray(base_vel_body),
            yaw=np.asarray(yaw_arr),
            feet_z=None if feet_z is None else np.asarray(feet_z),
            feet_z_tar=np.asarray(jnp.array(z_feet_tars)) if len(z_feet_tars) == T_total else None,
            joint_angles=np.asarray(joint_angles),
            controls=np.asarray(us_arr),
            vel_tar=np.asarray(vel_tar_series),
            ang_vel_tar=np.asarray(ang_vel_tar_series),
            algo=optimizer_display,
            env_name=dial_config.env_name,
        )
        print(f"Saved rollout metrics to {os.path.join(dial_config.output_dir, f'{timestamp}_rollout_metrics.npz')}")
    except Exception as e:
        print(f"Could not save rollout metrics: {e}")

    # Save learning curve data for sample efficiency analysis
    try:
        step_numbers = np.arange(len(rews))
        step_rewards = np.array(rews)
        samples_per_step = dial_config.Nsample
        cumulative_samples = (step_numbers + 1) * samples_per_step
        
        # Compute running average reward for smoother curves
        window_size = max(5, min(50, len(step_rewards) // 4))  # Adaptive window size with a floor
        weights = np.ones(window_size, dtype=float)
        running_sum = np.convolve(step_rewards, weights, mode='same')
        normalizer = np.convolve(np.ones_like(step_rewards, dtype=float), weights, mode='same')
        running_avg_rewards = running_sum / np.maximum(normalizer, 1e-8)
        running_avg_samples = cumulative_samples
        running_avg_steps = step_numbers
        
        np.savez(
            os.path.join(dial_config.output_dir, f"{timestamp}_learning_curve.npz"),
            step_numbers=step_numbers,
            step_rewards=step_rewards,
            cumulative_samples=cumulative_samples,
            running_avg_rewards=running_avg_rewards,
            running_avg_samples=running_avg_samples,
            running_avg_steps=running_avg_steps,
            final_reward=float(step_rewards[-10:].mean()),  # Average of last 10 steps
            total_samples=int(cumulative_samples[-1]),
            samples_per_step=samples_per_step,
            algo=optimizer_display,
            env_name=dial_config.env_name,
        )
        print(f"Saved learning curve data to {os.path.join(dial_config.output_dir, f'{timestamp}_learning_curve.npz')}")
    except Exception as e:
        print(f"Could not save learning curve data: {e}")

    # Save convergence traces for per-step inner-iteration analysis
    try:
        if len(convergence_traces) > 0:
            max_len = max(len(tr) for tr in convergence_traces)
            traces_padded = np.full((len(convergence_traces), max_len), np.nan, dtype=float)
            lengths = np.zeros((len(convergence_traces),), dtype=int)
            for i, tr in enumerate(convergence_traces):
                tr_np = np.asarray(tr, dtype=float)
                L = tr_np.shape[0]
                lengths[i] = L
                traces_padded[i, :L] = tr_np
            mean_trace = np.nanmean(traces_padded, axis=0)
            iters = np.arange(1, max_len + 1)
            np.savez(
                os.path.join(dial_config.output_dir, f"{timestamp}_convergence_traces.npz"),
                traces=traces_padded,
                lengths=lengths,
                mean_trace=mean_trace,
                iters=iters,
                algo=optimizer_display,
                env_name=dial_config.env_name,
            )
            print(f"Saved convergence traces to {os.path.join(dial_config.output_dir, f'{timestamp}_convergence_traces.npz')}")
    except Exception as e:
        print(f"Could not save convergence traces: {e}")

    @app.route("/")
    def index():
        return webpage

    app.run(port=args.port if args.port is not None else 5000)


if __name__ == "__main__":
    main()

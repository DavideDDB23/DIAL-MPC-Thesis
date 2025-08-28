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

plt.style.use("science")

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
    # Use the environment torso index (as in the env) for body-frame velocity/yaw
    torso_idx = getattr(env, "_torso_idx", 1) - 1
    # Compute walk-tracking error only for the unitree_go2_trot example
    compute_walk_tracking = (args.example == "unitree_go2_trot")

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

    @app.route("/")
    def index():
        return webpage

    app.run(port=args.port if args.port is not None else 5000)


if __name__ == "__main__":
    main()

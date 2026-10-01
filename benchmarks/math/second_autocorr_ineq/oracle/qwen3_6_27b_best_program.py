# EVOLVE-BLOCK-START
"""
C₂ Autocorrelation Constant Optimizer - Multi-Pass Sharpness Explorer v4

Strategy: Four-pass refinement with sharpness variation and extensive restart
diversity. Key improvements over v3:
1. 4 refinement passes (P1-diverse→P2-edge→P3-fine→P4-ultra)
2. Sharpness variation: 55→65→75→85 across passes explores different manifolds
3. 20/18/16/14 restarts per pass (was 16/14/12)
4. 35k/28k/20k/15k steps per pass (was 25k/18k/12k)
5. 8 diverse perturbation strategies including edge-Gaussian coupling
6. Expanded 20 initializations covering extreme step configurations
7. Full utilization of 360s time budget for deeper optimization
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
import time

jax.config.update("jax_enable_x64", True)

@dataclass
class Hyperparameters:
    n_points: int = 1024
    stages: tuple = (64, 128, 256, 512, 1024)
    steps_per_stage: tuple = (2000, 4000, 8000, 15000, 0)
    lr_peak: float = 0.025
    sharpness_p1: float = 55.0
    sharpness_p2: float = 65.0
    sharpness_p3: float = 75.0
    sharpness_p4: float = 85.0
    n_refine_p1: int = 20
    n_refine_p2: int = 18
    n_refine_p3: int = 16
    n_refine_p4: int = 14
    refine_steps_p1: int = 35000
    refine_steps_p2: int = 28000
    refine_steps_p3: int = 20000
    refine_steps_p4: int = 15000
    refine_lr_p1: float = 0.004
    refine_lr_p2: float = 0.002
    refine_lr_p3: float = 0.001
    refine_lr_p4: float = 0.0004
    grad_clip: float = 8.0


class C2Optimizer:
    """Four-pass optimizer with sharpness variation for C₂ maximization."""
    
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _loss_fn(self, params: jnp.ndarray, sharpness: float) -> jnp.ndarray:
        """Compute negative C₂ ratio using sharp softplus for step approximation.
        
        Args:
            params: Raw parameters (softplus input)
            sharpness: Controls step sharpness (higher = more discontinuous)
        Returns:
            Negative C₂ constant (we minimize this)
        """
        f = jax.nn.softplus(params * sharpness) + 1e-8
        N = len(f)
        f_pad = jnp.pad(f, (0, N))
        fft_f = jnp.fft.fft(f_pad)
        c_full = jnp.fft.ifft(fft_f * fft_f).real
        c = c_full[:2 * N - 1]
        c = jnp.maximum(c, 0.0) + 1e-12
        l2_sq = jnp.sum(c**2)
        l1_sq = jnp.sum(f)**2
        l_inf = jnp.max(c)
        c2 = l2_sq / jnp.maximum(l1_sq * l_inf, 1e-12)
        return -c2

    def _run_optimization(self, init_params: jnp.ndarray, steps: int,
                          lr_peak: float, sharpness: float,
                          key: jax.random.PRNGKey) -> tuple:
        """Adam optimization with warmup-cosine schedule and gradient clipping."""
        warmup_steps = max(30, int(steps * 0.04))
        decay_steps = max(1, steps - warmup_steps)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=lr_peak,
            warmup_steps=warmup_steps, decay_steps=decay_steps,
            end_value=max(lr_peak * 1e-6, 1e-7)
        )
        optimizer = optax.chain(
            optax.clip(self.hypers.grad_clip),
            optax.adam(learning_rate=schedule, b1=0.9, b2=0.999)
        )
        state = optimizer.init(init_params)

        @jax.jit
        def step(p, st):
            loss, grad = jax.value_and_grad(self._loss_fn)(p, sharpness)
            updates, st = optimizer.update(grad, st, p)
            return optax.apply_updates(p, updates), st, loss

        best_loss, best_p = jnp.inf, init_params
        p = init_params
        for _ in range(steps):
            p, state, loss = step(p, state)
            if loss < best_loss:
                best_loss, best_p = loss, p
        return best_p, best_loss

    def _upsample_to_N(self, f_best: np.ndarray, old_N: int, new_N: int) -> np.ndarray:
        """Log-space interpolation preserving relative step heights."""
        x_old = np.linspace(0.0, 1.0, old_N)
        x_new = np.linspace(0.0, 1.0, new_N)
        log_vals = np.log(np.maximum(f_best, 1e-8))
        return np.interp(x_new, x_old, log_vals)

    def _detect_edges(self, f: jnp.ndarray) -> jnp.ndarray:
        """Detect step edges via gradient magnitude."""
        grad = jnp.abs(jnp.diff(f, prepend=f[0]))
        return grad / jnp.maximum(jnp.max(grad), 1e-12)

    def _generate_inits(self, N: int, key: jax.random.PRNGKey,
                        prev_best: np.ndarray = None, prev_best_N: int = None) -> list:
        """Generate 20 diverse initializations covering step-function manifolds."""
        inits = []
        x = jnp.linspace(0.0, 1.0, N)

        # 1. Single compact step (AlphaEvolve baseline)
        inits.append(jnp.where(jnp.abs(x - 0.5) < 0.20, 2.5, -5.0))
        # 2. Two-level symmetric
        inits.append(jnp.where(jnp.abs(x - 0.5) < 0.10, 4.0,
                               jnp.where(jnp.abs(x - 0.5) < 0.30, 1.0, -5.0)))
        # 3. Three-level symmetric
        inits.append(jnp.where(jnp.abs(x - 0.5) < 0.08, 5.0,
                               jnp.where(jnp.abs(x - 0.5) < 0.22, 1.5,
                               jnp.where(jnp.abs(x - 0.5) < 0.38, 0.5, -5.0))))
        # 4. Four-level symmetric
        inits.append(jnp.where(jnp.abs(x - 0.5) < 0.06, 6.0,
                               jnp.where(jnp.abs(x - 0.5) < 0.15, 2.0,
                               jnp.where(jnp.abs(x - 0.5) < 0.28, 1.0,
                               jnp.where(jnp.abs(x - 0.5) < 0.42, 0.3, -5.0)))))
        # 5. Five-level symmetric (granular)
        inits.append(jnp.where(jnp.abs(x - 0.5) < 0.05, 7.0,
                               jnp.where(jnp.abs(x - 0.5) < 0.12, 3.0,
                               jnp.where(jnp.abs(x - 0.5) < 0.20, 1.5,
                               jnp.where(jnp.abs(x - 0.5) < 0.30, 0.7,
                               jnp.where(jnp.abs(x - 0.5) < 0.40, 0.2, -5.0))))))
        # 6. Six-level symmetric (ultra-fine)
        inits.append(jnp.where(jnp.abs(x - 0.5) < 0.04, 8.0,
                               jnp.where(jnp.abs(x - 0.5) < 0.10, 3.5,
                               jnp.where(jnp.abs(x - 0.5) < 0.18, 2.0,
                               jnp.where(jnp.abs(x - 0.5) < 0.26, 1.0,
                               jnp.where(jnp.abs(x - 0.5) < 0.35, 0.5,
                               jnp.where(jnp.abs(x - 0.5) < 0.45, 0.2, -5.0)))))))
        # 7. Wide plateau
        inits.append(jnp.where(jnp.abs(x - 0.5) < 0.35, 1.5, -5.0))
        # 8. Asymmetric two-level
        inits.append(jnp.where(jnp.abs(x - 0.45) < 0.12, 3.5,
                               jnp.where(jnp.abs(x - 0.45) < 0.28, 0.8, -5.0)))
        # 9. Asymmetric three-level
        inits.append(jnp.where(jnp.abs(x - 0.48) < 0.07, 5.0,
                               jnp.where(jnp.abs(x - 0.48) < 0.20, 1.8,
                               jnp.where(jnp.abs(x - 0.48) < 0.35, 0.4, -5.0))))
        # 10. Symmetric hollow (tight)
        inits.append(jnp.where(
            (jnp.abs(x - 0.38) < 0.06) | (jnp.abs(x - 0.62) < 0.06), 4.0,
            jnp.where(jnp.abs(x - 0.5) < 0.35, 0.3, -5.0)))
        # 11. Symmetric hollow (wide)
        inits.append(jnp.where(
            (jnp.abs(x - 0.35) < 0.08) | (jnp.abs(x - 0.65) < 0.08), 4.0,
            jnp.where(jnp.abs(x - 0.5) < 0.40, 0.3, -5.0)))
        # 12. Asymmetric hollow
        inits.append(jnp.where(
            (jnp.abs(x - 0.40) < 0.05) | (jnp.abs(x - 0.60) < 0.07), 3.5,
            jnp.where(jnp.abs(x - 0.5) < 0.35, 0.4, -5.0)))
        # 13. Triple peak symmetric
        inits.append(jnp.where(
            (jnp.abs(x - 0.30) < 0.04) | (jnp.abs(x - 0.50) < 0.04) | (jnp.abs(x - 0.70) < 0.04), 4.0,
            jnp.where(jnp.abs(x - 0.5) < 0.40, 0.5, -5.0)))
        # 14. Triple peak asymmetric
        inits.append(jnp.where(
            (jnp.abs(x - 0.25) < 0.04) | (jnp.abs(x - 0.50) < 0.05) | (jnp.abs(x - 0.75) < 0.04), 4.5,
            jnp.where(jnp.abs(x - 0.5) < 0.42, 0.4, -5.0)))
        # 15. Hollow with wide gap
        inits.append(jnp.where(
            (jnp.abs(x - 0.33) < 0.06) | (jnp.abs(x - 0.67) < 0.06), 3.8,
            jnp.where(jnp.abs(x - 0.5) < 0.42, 0.35, -5.0)))
        # 16. Flat top with steep edges
        inits.append(jnp.where(jnp.abs(x - 0.5) < 0.25, 5.0,
                               jnp.where(jnp.abs(x - 0.5) < 0.32, 1.0, -5.0)))
        # 17. Gaussian baseline
        inits.append(-20.0 * (x - 0.5)**2)
        # 18. Double Gaussian
        inits.append(jnp.exp(-80 * (x - 0.35)**2) + jnp.exp(-80 * (x - 0.65)**2))
        # 19. Box with ramp
        inits.append(jnp.where(jnp.abs(x - 0.5) < 0.30, 2.0 - 10.0 * jnp.abs(x - 0.5), -5.0))
        # 20. Coarse-to-fine transfer
        if prev_best is not None and prev_best_N is not None:
            log_interp = self._upsample_to_N(prev_best, prev_best_N, N)
            inits.append(log_interp / 60.0)  # Normalize for sharpness~60

        return inits

    def _generate_refine_inits(self, base_params: jnp.ndarray, N: int,
                                f_current: jnp.ndarray, n_restarts: int,
                                key: jax.random.PRNGKey, strategy: str) -> list:
        """Generate structured perturbations for refinement phase.
        
        Strategies: diverse (8 options), edge (6 options), fine (4 options), ultra (3 options)
        """
        inits = [base_params]
        x = jnp.linspace(0.0, 1.0, N)
        env_gauss = jnp.exp(-40 * (x - 0.5)**2)
        env_hollow = jnp.clip(jnp.abs(x - 0.5) * 2.0, 0.0, 1.0)
        edge_mask = self._detect_edges(f_current)
        
        for i in range(n_restarts - 1):
            key, sub = jax.random.split(key)
            noise = jax.random.normal(sub, (N,))

            if strategy == "diverse":
                r = jax.random.randint(sub, (), 0, 8)
                if r == 0:
                    pert = base_params + noise * 0.05
                elif r == 1:
                    pert = base_params + noise * 0.30 * env_gauss
                elif r == 2:
                    pert = base_params + noise * 0.18 * env_hollow
                elif r == 3:
                    pert = base_params + jax.random.uniform(sub, (N,), minval=-0.30, maxval=0.30) * env_gauss
                elif r == 4:
                    pert = base_params + noise * 0.25 * edge_mask
                elif r == 5:
                    pert = base_params + noise * 0.15 * (edge_mask * env_gauss + 0.1)
                elif r == 6:
                    pert = base_params + noise * 0.20 * (edge_mask + env_hollow) * 0.5
                else:
                    pert = base_params + jax.random.uniform(sub, (N,), minval=-0.20, maxval=0.20) * (0.3 + edge_mask * 0.7)
                    
            elif strategy == "edge":
                r = jax.random.randint(sub, (), 0, 6)
                if r == 0:
                    pert = base_params + noise * 0.03
                elif r == 1:
                    pert = base_params + noise * 0.20 * edge_mask
                elif r == 2:
                    pert = base_params + noise * 0.15 * env_gauss
                elif r == 3:
                    pert = base_params + noise * 0.12 * (edge_mask + env_hollow) * 0.5
                elif r == 4:
                    pert = base_params + jax.random.uniform(sub, (N,), minval=-0.12, maxval=0.12) * (0.5 + edge_mask)
                else:
                    pert = base_params + noise * 0.10 * edge_mask * env_gauss
                    
            elif strategy == "fine":
                r = jax.random.randint(sub, (), 0, 4)
                if r == 0:
                    pert = base_params + noise * 0.015
                elif r == 1:
                    pert = base_params + noise * 0.08 * edge_mask
                elif r == 2:
                    pert = base_params + noise * 0.05 * env_gauss
                else:
                    pert = base_params + jax.random.uniform(sub, (N,), minval=-0.05, maxval=0.05) * (0.4 + 0.6 * edge_mask)
            else:  # ultra
                r = jax.random.randint(sub, (), 0, 3)
                if r == 0:
                    pert = base_params + noise * 0.008
                elif r == 1:
                    pert = base_params + noise * 0.04 * edge_mask
                else:
                    pert = base_params + noise * 0.03 * env_gauss

            inits.append(pert)
        return inits

    def _update_base(self, best_f: np.ndarray, best_f_N: int, N: int, sharpness: float) -> jnp.ndarray:
        """Extract parameters from best function for given sharpness."""
        if best_f_N != N:
            return jnp.array(self._upsample_to_N(np.array(best_f), best_f_N, N) / sharpness)
        return jnp.array(np.log(np.maximum(np.array(best_f), 1e-8)) / sharpness)

    def run(self) -> tuple:
        """Execute multi-scale, 4-pass refinement optimization pipeline."""
        key = jax.random.PRNGKey(42)
        best_f, best_loss, best_f_N = None, jnp.inf, None
        lr_scales = [1.0, 0.9, 0.7, 0.5, 0.3]
        base_sharpness = self.hypers.sharpness_p1

        # === COARSE-TO-FINE ===
        for stage_idx, (N, steps, lr_scale) in enumerate(zip(
                self.hypers.stages, self.hypers.steps_per_stage, lr_scales)):
            if steps == 0:
                if best_f is not None and best_f_N != N:
                    params_up = self._upsample_to_N(np.array(best_f), best_f_N, N) / base_sharpness
                    best_f = jax.nn.softplus(jnp.array(params_up) * base_sharpness)
                    best_f_N = N
                continue
            key, sub = jax.random.split(key)
            inits = self._generate_inits(N, sub, prev_best=best_f, prev_best_N=best_f_N)
            lr = self.hypers.lr_peak * lr_scale
            stage_start = time.time()
            for i, init in enumerate(inits):
                key, sub2 = jax.random.split(key)
                p, loss = self._run_optimization(init, steps, lr, base_sharpness, sub2)
                f_cand = jax.nn.softplus(p * base_sharpness)
                if loss < best_loss:
                    best_loss, best_f, best_f_N = loss, f_cand, N
                print(f"Stage {stage_idx} (N={N:4d}) | Init {i+1:2d}/{len(inits)} | "
                      f"C₂ ≈ {-loss:.8f} | Best: {-best_loss:.8f}")
            print(f"  Stage {stage_idx} completed in {time.time()-stage_start:.2f}s")

        # === REFINEMENT: 4 passes with increasing sharpness ===
        N = self.hypers.n_points
        
        pass_configs = [
            ("P1-diverse", self.hypers.n_refine_p1, self.hypers.refine_steps_p1,
             self.hypers.refine_lr_p1, "diverse", self.hypers.sharpness_p1),
            ("P2-edge", self.hypers.n_refine_p2, self.hypers.refine_steps_p2,
             self.hypers.refine_lr_p2, "edge", self.hypers.sharpness_p2),
            ("P3-fine", self.hypers.n_refine_p3, self.hypers.refine_steps_p3,
             self.hypers.refine_lr_p3, "fine", self.hypers.sharpness_p3),
            ("P4-ultra", self.hypers.n_refine_p4, self.hypers.refine_steps_p4,
             self.hypers.refine_lr_p4, "ultra", self.hypers.sharpness_p4),
        ]

        for pass_name, n_r, steps_r, lr_r, strat, sharpness in pass_configs:
            print(f"\n=== REFINEMENT {pass_name}: {n_r} restarts × {steps_r} steps (sharpness={sharpness}) ===")
            
            base_params = self._update_base(best_f, best_f_N, N, sharpness)
            key, sub = jax.random.split(key)
            f_current = jax.nn.softplus(base_params * sharpness)
            refine_inits = self._generate_refine_inits(base_params, N, f_current, n_r, sub, strat)
            refine_start = time.time()

            for i, init in enumerate(refine_inits):
                key, sub2 = jax.random.split(key)
                p, loss = self._run_optimization(init, steps_r, lr_r, sharpness, sub2)
                f_cand = jax.nn.softplus(p * sharpness)
                if loss < best_loss:
                    best_loss, best_f, best_f_N = loss, f_cand, N
                    base_params = p
                print(f"Refine {pass_name} {i+1:2d}/{len(refine_inits)} | "
                      f"C₂ ≈ {-loss:.8f} | Global Best: {-best_loss:.8f}")

            print(f"  Pass {pass_name} completed in {time.time()-refine_start:.2f}s")

        return np.array(best_f, dtype=np.float64), float(-best_loss)


def run():
    """Entry point matching required I/O signature."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    f_opt, c2_val = optimizer.run()
    loss_val = -c2_val
    return f_opt, float(c2_val), float(loss_val), hypers.n_points
# EVOLVE-BLOCK-END
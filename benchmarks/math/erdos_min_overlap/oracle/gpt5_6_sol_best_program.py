# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

# Cold log-sum-exp continuation and simplex projection need more precision than
# float32 provides when active correlations differ by roughly 1e-7.
jax.config.update("jax_enable_x64", True)


@dataclass
class Hyperparameters:
    num_intervals: int = 200
    learning_rate: float = 0.012
    num_steps: int = 32000
    num_restarts: int = 24
    start_temperature: float = 0.003
    end_temperature: float = 0.00002


class ErdosOptimizer:
    """
    Finds a step function h that minimizes the maximum overlap integral.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 2.0
        self.dx = self.domain_width / self.hypers.num_intervals

    def _feasible_h(self, latent_h_values: jnp.ndarray) -> jnp.ndarray:
        """Map latent values into [0,1] while making their mean exactly one half."""
        h = jax.nn.sigmoid(latent_h_values)
        mean = jnp.mean(h)
        lower = h / (2.0 * mean)
        upper = 1.0 - (1.0 - h) / (2.0 * (1.0 - mean))
        return jnp.where(mean >= 0.5, lower, upper)

    def _correlations(self, h: jnp.ndarray) -> jnp.ndarray:
        """Evaluate all zero-extended discrete-shift overlap integrals by FFT."""
        N = self.hypers.num_intervals
        hp = jnp.pad(h, (0, N))
        jp = jnp.pad(1.0 - h, (0, N))
        spectrum = jnp.fft.fft(hp) * jnp.conj(jnp.fft.fft(jp))
        return jnp.fft.ifft(spectrum).real * self.dx

    def _objective_fn(
        self, latent_h_values: jnp.ndarray, temperature: jnp.ndarray
    ) -> jnp.ndarray:
        """Minimize an annealed log-sum-exp approximation to the exact maximum."""
        correlations = self._correlations(self._feasible_h(latent_h_values))
        return temperature * jax.scipy.special.logsumexp(
            correlations / temperature
        )

    def run_optimization(self):
        """Anneal random and symmetry-informed starts, then polish exact minimax basins."""
        schedule = optax.cosine_decay_schedule(
            self.hypers.learning_rate, self.hypers.num_steps, alpha=0.015
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(schedule)
        )
        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals

        # Combine high-frequency, symmetry-informed, and coarse multiscale
        # starts so continuation explores structurally different basins.
        raw = jax.random.normal(key, (12, N))
        raw *= jnp.linspace(0.35, 2.5, 12)[:, None]

        source = jax.random.normal(jax.random.fold_in(key, 1), (2, N))
        reflected = 0.5 * (source - source[:, ::-1])
        reflected *= jnp.array([0.8, 2.0])[:, None]

        half = jax.random.normal(jax.random.fold_in(key, 2), (2, N // 2))
        shifted = jnp.concatenate((half, -half), axis=1)
        shifted *= jnp.array([0.8, 2.0])[:, None]

        coarse = jax.random.normal(
            jax.random.fold_in(key, 3), (8, N // 5)
        )
        smooth = jnp.repeat(coarse, 5, axis=1)
        smooth -= jnp.mean(smooth, axis=1, keepdims=True)
        smooth *= jnp.linspace(0.45, 2.4, 8)[:, None]

        latent_h_values = jnp.concatenate(
            (raw, reflected, shifted, smooth), axis=0
        )
        opt_state = optimizer.init(latent_h_values)

        @jax.jit
        def train_step(latent_h_values, opt_state, temperature):
            """Apply one clipped Adam step at the current continuation temperature."""
            def batch_loss(values):
                losses = jax.vmap(
                    lambda z: self._objective_fn(z, temperature)
                )(values)
                return jnp.sum(losses)

            grads = jax.grad(batch_loss)(latent_h_values)
            updates, opt_state = optimizer.update(
                grads, opt_state, latent_h_values
            )
            return optax.apply_updates(latent_h_values, updates), opt_state

        ratio = self.hypers.end_temperature / self.hypers.start_temperature
        denominator = max(self.hypers.num_steps - 1, 1)
        for step in range(self.hypers.num_steps):
            temperature = self.hypers.start_temperature * (
                ratio ** (step / denominator)
            )
            latent_h_values, opt_state = train_step(
                latent_h_values, opt_state, temperature
            )

        candidates = jax.vmap(self._feasible_h)(latent_h_values)
        bounds = jax.vmap(
            lambda h: jnp.max(self._correlations(h))
        )(candidates)

        # Polish strong basins directly in h-space. This avoids the vanishing
        # sigmoid derivatives that impede accurate active-constraint balancing.
        elite = candidates[jnp.argsort(bounds)[:6]]
        polish_steps = 8000
        polish_schedule = optax.cosine_decay_schedule(
            4e-4, polish_steps, alpha=0.01
        )
        polish_optimizer = optax.adam(polish_schedule)
        polish_state = polish_optimizer.init(elite)

        def project_capped_simplex(values):
            """Project one vector onto [0,1]^N with sum exactly N/2."""
            low = jnp.min(values) - 1.0
            high = jnp.max(values)
            for _ in range(32):
                midpoint = 0.5 * (low + high)
                total = jnp.mean(jnp.clip(values - midpoint, 0.0, 1.0))
                low, high = jax.lax.cond(
                    total > 0.5,
                    lambda _: (midpoint, high),
                    lambda _: (low, midpoint),
                    operand=None,
                )
            projected = jnp.clip(values - 0.5 * (low + high), 0.0, 1.0)
            residual = 0.5 * projected.size - jnp.sum(projected)
            index = jnp.argmax(jnp.minimum(projected, 1.0 - projected))
            return projected.at[index].add(residual)

        def direct_loss(values, temperature):
            """Minimize a low-temperature smooth maximum directly over feasible h."""
            correlations = self._correlations(values)
            return temperature * jax.scipy.special.logsumexp(
                correlations / temperature
            )

        @jax.jit
        def polish_step(values, state, temperature):
            """Take projected Adam steps that equalize the largest correlations."""
            loss = lambda batch: jnp.sum(jax.vmap(
                lambda h: direct_loss(h, temperature)
            )(batch))
            grads = jax.grad(loss)(values)
            updates, state = polish_optimizer.update(grads, state, values)
            values = optax.apply_updates(values, updates)
            values = jax.vmap(project_capped_simplex)(values)
            return values, state

        polish_start_temperature = self.hypers.end_temperature
        polish_end_temperature = 5e-7
        polish_ratio = polish_end_temperature / polish_start_temperature
        for step in range(polish_steps):
            temperature = polish_start_temperature * (
                polish_ratio ** (step / max(polish_steps - 1, 1))
            )
            elite, polish_state = polish_step(
                elite, polish_state, temperature
            )

        # Keep the pre-polish candidates as safeguards against a late minimax
        # oscillation, and select using the exact discrete maximum.
        candidates = jnp.concatenate((candidates, elite), axis=0)
        bounds = jax.vmap(
            lambda h: jnp.max(self._correlations(h))
        )(candidates)
        best = jnp.argmin(bounds)
        final_h = np.asarray(candidates[best], dtype=np.float64)
        c5_bound = float(bounds[best])

        # Solve the true finite minimax problem from the strongest JAX basin.
        # SLSQP sees an epigraph variable and exact overlap Jacobians, avoiding
        # the residual bias and ill-conditioning of very cold log-sum-exp.
        N = self.hypers.num_intervals
        L = 2 * N
        shifts = np.arange(L)[:, None]
        coordinates = np.arange(N)[None, :]
        left = (coordinates - shifts) % L
        right = (coordinates + shifts) % L
        left_valid = left < N
        right_valid = right < N
        left_index = left % N
        right_index = right % N

        def exact_correlations(h):
            """Evaluate all zero-extended grid-shift overlaps in float64."""
            hp = np.pad(h, (0, N))
            qp = np.pad(1.0 - h, (0, N))
            return (
                np.fft.ifft(np.fft.fft(hp) * np.conj(np.fft.fft(qp))).real
                * self.dx
            )

        def correlation_jacobian(h):
            """Return the analytic Jacobian of every overlap with respect to h."""
            first = np.where(left_valid, 1.0 - h[left_index], 0.0)
            second = np.where(right_valid, h[right_index], 0.0)
            return self.dx * (first - second)

        def inequalities(z):
            """Impose the exact epigraph inequalities t minus overlap(h)."""
            return z[-1] - exact_correlations(z[:-1])

        def inequality_jacobian(z):
            """Return analytic derivatives of all epigraph inequalities."""
            jac = -correlation_jacobian(z[:-1])
            return np.column_stack((jac, np.ones(L)))

        def project_numpy(values):
            """Project onto the capped simplex with mean exactly one half."""
            low = np.min(values) - 1.0
            high = np.max(values)
            for _ in range(60):
                midpoint = 0.5 * (low + high)
                if np.mean(np.clip(values - midpoint, 0.0, 1.0)) > 0.5:
                    low = midpoint
                else:
                    high = midpoint
            projected = np.clip(values - 0.5 * (low + high), 0.0, 1.0)
            residual = 0.5 * N - np.sum(projected)
            index = np.argmax(np.minimum(projected, 1.0 - projected))
            projected[index] += residual
            return projected

        objective_jacobian = np.r_[np.zeros(N), 1.0]
        equality_jacobian = np.r_[np.ones(N), 0.0]
        constraints = [
            {
                "type": "eq",
                "fun": lambda z: np.sum(z[:-1]) - 0.5 * N,
                "jac": lambda z: equality_jacobian,
            },
            {
                "type": "ineq",
                "fun": inequalities,
                "jac": inequality_jacobian,
            },
        ]

        # Reverify the incumbent in NumPy float64. Refine the globally best
        # candidate plus strong unpolished starts, rather than accidentally
        # spending every SLSQP run on polished copies of one basin.
        final_h = project_numpy(final_h)
        c5_bound = float(np.max(exact_correlations(final_h)))
        bounds_numpy = np.asarray(bounds, dtype=np.float64)
        global_best = int(np.argmin(bounds_numpy))
        original_order = np.argsort(
            bounds_numpy[:self.hypers.num_restarts]
        )[:8]
        seed_indices = [global_best]
        seed_indices.extend(
            int(index)
            for index in original_order
            if int(index) != global_best
        )
        seed_pool = np.asarray(candidates, dtype=np.float64)[seed_indices]
        for seed in seed_pool:
            seed = project_numpy(seed)
            seed_bound = float(np.max(exact_correlations(seed)))
            initial = np.r_[seed, seed_bound + 1e-10]
            result = minimize(
                lambda z: z[-1],
                initial,
                jac=lambda z: objective_jacobian,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * N + [(0.0, 1.0)],
                constraints=constraints,
                options={"maxiter": 600, "ftol": 1e-13, "disp": False},
            )

            if np.all(np.isfinite(result.x)):
                refined_h = project_numpy(result.x[:-1])
                refined_bound = float(np.max(exact_correlations(refined_h)))
                if refined_bound < c5_bound:
                    final_h = refined_h
                    c5_bound = refined_bound

        return final_h, c5_bound


def run():
    """Construct and optimize an exactly normalized 200-interval step function."""
    hypers = Hyperparameters()
    optimizer = ErdosOptimizer(hypers)
    final_h_values, c5_bound = optimizer.run_optimization()
    return final_h_values, c5_bound, hypers.num_intervals


# EVOLVE-BLOCK-END

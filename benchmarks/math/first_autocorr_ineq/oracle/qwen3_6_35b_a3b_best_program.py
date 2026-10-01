# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np

def run():
    """
    Minimizes max(f★f)/(∫f)² via log-parametrization, multi-stage smooth-max annealing,
    and robust multi-restart optimization.
    Improvements:
    1. Higher resolution (N=1200) and extended optimization (120k steps/restart) for finer peak resolution.
    2. Three-stage k-annealing (30→300→1500→4000) to balance global exploration and local precision.
    3. JAX stop-gradient on the peak value to stabilize smooth-max gradients and prevent oscillation.
    4. Log-sum-exp normalization for strict non-negativity, scale invariance, and numerical stability.
    5. Symmetry projection and diverse theoretical initializations (quartic, cosine, flat-top) to escape shallow minima.
    """
    N = 1200
    dx = 0.5 / N
    steps = 120000
    warmup = 4000
    lr = 0.02
    best_c1 = jnp.inf
    best_f = None
    
    x = jnp.linspace(-0.25, 0.25, N)
    
    # Multi-stage k schedule for progressive sharpening without gradient instability
    k1 = jnp.linspace(30.0, 300.0, steps // 3)
    k2 = jnp.linspace(300.0, 1500.0, steps // 3)
    k3 = jnp.linspace(1500.0, 4000.0, steps - 2*(steps//3))
    k_schedule = jnp.concatenate([k1, k2, k3])
    
    def objective(g, k):
        # Log-sum-exp normalization ensures scale invariance and strict positivity
        g_shifted = g - jnp.max(g)
        f = jnp.exp(g_shifted)
        I = jnp.maximum(jnp.sum(f) * dx, 1e-12)
        f_norm = f / I
        
        # FFT-based autoconvolution (linear convolution via zero-padding)
        pad = jnp.pad(f_norm, (0, N))
        conv = jnp.fft.ifft(jnp.fft.fft(pad)**2).real * dx
        
        # Smooth maximum with stop-gradient on peak to prevent gradient oscillation
        m = jnp.max(conv)
        diff = k * (conv - jax.lax.stop_gradient(m))
        return m + jnp.log(jnp.sum(jnp.exp(diff)) + 1e-12) / k

    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps - warmup, lr * 1e-4)
    optimizer = optax.adam(learning_rate=sched)
    
    @jax.jit
    def step(g, state, k):
        l, grads = jax.value_and_grad(objective)(g, k)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(g, updates), state, l
        
    for restart in range(6):
        key = jax.random.PRNGKey(restart * 137 + 42)
        init_type = restart % 4
        
        if init_type == 0:
            # Quartic bump (theoretical candidate for smooth autoconvolution)
            g = -((x / 0.15)**4) * 3.0
        elif init_type == 1:
            # Cosine-squared
            g = jnp.log(jnp.cos(jnp.pi * x / 0.5)**2 + 1e-6) * 1.5
        elif init_type == 2:
            # Flat-top with smooth edges
            g = jnp.where(jnp.abs(x) < 0.20, 1.2, -jnp.abs(x)*10.0)
        else:
            # Perturbed random
            g = jax.random.normal(key, (N,)) * 0.2
            
        # Enforce even symmetry to halve effective search space and match expected optimal shape
        g = (g + g[::-1]) / 2.0
            
        state = optimizer.init(g)
        
        for i in range(steps):
            g, state, _ = step(g, state, k_schedule[i])
            
        # Exact evaluation with standard max
        g_eval = g - jnp.max(g)
        f = jnp.exp(g_eval)
        I = jnp.sum(f) * dx
        f_norm = f / I
        pad = jnp.pad(f_norm, (0, N))
        conv = jnp.fft.ifft(jnp.fft.fft(pad)**2).real * dx
        c1 = float(jnp.max(conv))
        
        if c1 < best_c1:
            best_c1 = c1
            best_f = f
            
    return np.array(best_f), best_c1, best_c1, N

# EVOLVE-BLOCK-END

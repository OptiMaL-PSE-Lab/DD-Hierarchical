import numpy as np

def scale(x_unscaled, bounds):
    return np.array([(x_unscaled[i] - bounds[i][0])/(bounds[i][1] - bounds[i][0]) for i in range(len(x_unscaled))])
def unscale(x_scaled, bounds):
    return np.array([bounds[i][0] + x_scaled[i]*(bounds[i][1] - bounds[i][0]) for i in range(len(x_scaled))])

def optimizer(f, x0, bounds, N, radius):
    N_dim = len(x0)
    x_scaled = scale(x0, bounds)
    f_best = f(x0)
    x_low = np.random.uniform(
            low= [max(0, x_scaled[i] - radius) for i in range(N_dim)],
            high=[min(1, x_scaled[i] + radius) for i in range(N_dim)],
        )
    
    count = 1
    while count < N:
        f_ = f(unscale(x_low, bounds))
        if f_ < f_best:
            x_scaled = x_low
            f_best = f_
        count += 1
        x_low = np.random.uniform(
            low= [max(0, x_scaled[i] - radius) for i in range(N_dim)],
            high=[min(1, x_scaled[i] + radius) for i in range(N_dim)],
        )

    x_best = unscale(x_scaled, bounds)
    return f_best, x_best




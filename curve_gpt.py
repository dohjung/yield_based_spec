import numpy as np
import matplotlib.pyplot as plt

# --- Metric functions ---

def monotonicity_index(x, y):
    x = np.array(x)
    y = np.array(y)

    idx_sort = np.argsort(x)
    x = x[idx_sort]
    y = y[idx_sort]

    dy = np.diff(y)
    dx = np.diff(x)
    dx[dx == 0] = 1e-8

    slopes = dy / dx
    net_change = y[-1] - y[0]
    total_change = np.sum(np.abs(slopes * dx))

    if total_change == 0:
        return 1.0
    return abs(net_change) / total_change

def compute_skewness(x, y):
    x = np.array(x)
    y = np.array(y)

    idx_sort = np.argsort(x)
    x = x[idx_sort]
    y = y[idx_sort]

    dx = np.diff(x)
    y_mid = (y[:-1] + y[1:]) / 2

    weights = dx
    mean_y = np.average(y_mid, weights=weights)
    var_y = np.average((y_mid - mean_y) ** 2, weights=weights)
    std_y = np.sqrt(var_y)

    if std_y < 1e-8:
        return 0.0

    skew = np.average(((y_mid - mean_y) / std_y) ** 3, weights=weights)
    return skew

# --- Classifier ---

def classify_curve(x, y):
    x = np.array(x)
    y = np.array(y)
    n = len(y)
    if n < 2:
        return "Unknown"

    idx_sort = np.argsort(x)
    x = x[idx_sort]
    y = y[idx_sort]

    m_index = monotonicity_index(x, y)
    trend = y[-1] - y[0]
    if m_index > 0.5:
        if trend > 0:
            return "J-shape (increasing)"
        elif trend < 0:
            return "L-shape (decreasing)"
        else:
            return "Flat or random"

    one_fifth = max(1, n // 5)
    first_segment = np.mean(y[:one_fifth])
    last_segment = np.mean(y[-one_fifth:])
    middle_range = (x >= np.percentile(x, 33)) & (x <= np.percentile(x, 67))
    middle_segment = np.mean(y[middle_range])

    min_idx = int(np.argmin(y))
    max_idx = int(np.argmax(y))

    if 0 < min_idx < n - 1:
        if first_segment > middle_segment * 1.1 and last_segment > middle_segment * 1.1:
            return "U-shape (valley in middle)"
    if 0 < max_idx < n - 1:
        if first_segment < middle_segment * 0.9 and last_segment < middle_segment * 0.9:
            return "Inverted U-shape (peak in middle)"

    return "Random/Unclassified"

# --- Generate synthetic test curves ---

x = np.linspace(0, 100, 20)

def add_noise(y, level=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return y + level * rng.standard_normal(len(y))

y_U = 0.2 + 0.8 * ((x - 50) / 50) ** 2
y_invU = 1.0 - 0.8 * ((x - 50) / 50) ** 2
y_J = 0.1 + 0.9 * (x / 100) ** 2
y_L = 0.1 + 0.9 * np.exp(-5 * (x / 100))
y_rand = 0.5 + 0.2 * np.random.default_rng(1).standard_normal(len(x))

curves = {
    "U-shape": add_noise(y_U, seed=0),
    "Inverted U-shape": add_noise(y_invU, seed=1),
    "J-shape": add_noise(y_J, seed=2),
    "L-shape": add_noise(y_L, seed=3),
    "Random": y_rand,
}

# --- Plot and classify each curve ---

fig, axes = plt.subplots(len(curves), 1, figsize=(8, 12), sharex=True)

for ax, (name, y_vals) in zip(axes, curves.items()):
    shape = classify_curve(x, y_vals)
    m_idx = monotonicity_index(x, y_vals)
    skew = compute_skewness(x, y_vals)
    sym_corr = np.corrcoef(y_vals, y_vals[::-1])[0, 1]

    ax.plot(x, y_vals, marker='o')
    ax.set_title(f"{name}: {shape}\n"
                 f"Monotonicity={m_idx:.2f}, Skewness={skew:.2f}, Symmetry={sym_corr:.2f}")
    ax.grid(True)

plt.tight_layout()
plt.show()

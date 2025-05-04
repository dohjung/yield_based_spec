import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def generate_sample_curves(n_points=100, noise_level=0.05):
    """
    Generate sample curves for different defect rate patterns
    
    Args:
        n_points: Number of points to generate
        noise_level: Amount of noise to add to the curves
        
    Returns:
        Dictionary of measurement values and defect rates for each curve type
    """
    # Generate measurement values
    x = np.linspace(-5, 5, n_points)
    
    curves = {}
    
    # 1. U-shape curve
    y_u = 0.1 + 0.3 * x**2 + noise_level * np.random.randn(n_points)
    curves['U-shape'] = {'measurement': x, 'defect_rate': y_u}
    
    # 2. J-shape curve
    y_j = 0.1 + 0.5 * np.exp(0.8 * (x - 2)) / (1 + np.exp(0.8 * (x - 2))) + noise_level * np.random.randn(n_points)
    curves['J-shape'] = {'measurement': x, 'defect_rate': y_j}
    
    # 3. L-shape curve
    y_l = 0.1 + 0.5 * np.exp(-0.8 * (x + 2)) / (1 + np.exp(-0.8 * (x + 2))) + noise_level * np.random.randn(n_points)
    curves['L-shape'] = {'measurement': x, 'defect_rate': y_l}
    
    # 4. Inverted U-shape curve
    y_inv_u = 0.5 - 0.3 * x**2 + noise_level * np.random.randn(n_points)
    y_inv_u = np.clip(y_inv_u, 0.05, 1.0)  # Ensure non-negative defect rates
    curves['Inverted U-shape'] = {'measurement': x, 'defect_rate': y_inv_u}
    
    # 5. Random shape (no relationship)
    y_random = 0.2 + 0.1 * np.random.randn(n_points)
    y_random = np.clip(y_random, 0.05, 0.95)  # Bound between 0 and 1
    curves['Random'] = {'measurement': x, 'defect_rate': y_random}
    
    return curves


def compute_metrics(x, y):
    """
    Compute metrics for a given defect rate curve
    
    Args:
        x: Measurement values
        y: Defect rates
        
    Returns:
        Dictionary of metrics
    """
    # Normalize x and y to [0,1] for consistent analysis
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y)) if np.max(y) > np.min(y) else y
    
    # Fit a spline to smooth the data
    spline = UnivariateSpline(x_norm, y_norm, s=0.1)
    y_smooth = spline(x_norm)
    
    # Calculate derivatives
    dx = x_norm[1] - x_norm[0]
    dy_dx = np.gradient(y_smooth, dx)
    d2y_dx2 = np.gradient(dy_dx, dx)
    
    # Find peaks and valleys
    peaks, _ = find_peaks(y_smooth)
    valleys, _ = find_peaks(-y_smooth)
    
    # Calculate distribution metrics
    skewness = skew(y_norm)
    kurt = kurtosis(y_norm)
    
    # Calculate left, middle, and right averages to detect asymmetry
    n = len(y_norm)
    left_avg = np.mean(y_norm[:n//3])
    middle_avg = np.mean(y_norm[n//3:2*n//3])
    right_avg = np.mean(y_norm[2*n//3:])
    
    # Calculate quadratic fit - for U and inverted-U shapes
    X_poly = PolynomialFeatures(degree=2).fit_transform(x_norm.reshape(-1, 1))
    model = LinearRegression().fit(X_poly, y_norm)
    quad_coef = model.coef_[2]  # Coefficient of x^2
    
    # Calculate R-squared of quadratic fit
    y_pred = model.predict(X_poly)
    ss_total = np.sum((y_norm - np.mean(y_norm))**2)
    ss_residual = np.sum((y_norm - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Check if curve is largely monotonic in either direction
    monotonic_increasing = np.mean(dy_dx > 0) > 0.8
    monotonic_decreasing = np.mean(dy_dx < 0) > 0.8
    
    # Calculate edge-to-center ratio
    edge_avg = (left_avg + right_avg) / 2
    edge_to_center_ratio = edge_avg / middle_avg if middle_avg > 0 else float('inf')
    
    # Calculate variance in left and right halves
    left_var = np.var(y_norm[:n//2])
    right_var = np.var(y_norm[n//2:])
    var_ratio = left_var / right_var if right_var > 0 else float('inf')
    
    return {
        'skewness': skewness,
        'kurtosis': kurt,
        'num_peaks': len(peaks),
        'num_valleys': len(valleys),
        'left_avg': left_avg,
        'middle_avg': middle_avg,
        'right_avg': right_avg,
        'edge_to_center_ratio': edge_to_center_ratio,
        'quadratic_coefficient': quad_coef,
        'r_squared': r_squared,
        'monotonic_increasing': monotonic_increasing,
        'monotonic_decreasing': monotonic_decreasing,
        'var_ratio': var_ratio,
    }


def classify_curve_shape(metrics):
    """
    Classify the shape of a curve based on computed metrics
    
    Args:
        metrics: Dictionary of metrics computed from the curve
        
    Returns:
        String indicating the predicted curve shape
    """
    # Extract key metrics
    skewness = metrics['skewness']
    kurtosis = metrics['kurtosis']
    quad_coef = metrics['quadratic_coefficient']
    r_squared = metrics['r_squared']
    edge_to_center = metrics['edge_to_center_ratio']
    left_avg = metrics['left_avg']
    right_avg = metrics['right_avg']
    middle_avg = metrics['middle_avg']
    var_ratio = metrics['var_ratio']
    
    # Classification rules
    if r_squared < 0.3:
        return "Random shape", 0.8
    
    # U-shape characteristics
    if quad_coef > 0.5 and abs(skewness) < 0.5 and edge_to_center > 1.5:
        confidence = r_squared * 0.9  # Higher confidence with better quadratic fit
        return "U-shape", confidence
    
    # Inverted U-shape characteristics
    if quad_coef < -0.5 and abs(skewness) < 0.5 and edge_to_center < 0.7:
        confidence = r_squared * 0.9  # Higher confidence with better quadratic fit
        return "Inverted U-shape", confidence
    
    # J-shape characteristics (right side higher)
    if skewness > 0.8 and right_avg > left_avg * 2 and right_avg > middle_avg:
        confidence = 0.7 + 0.2 * (skewness / 2) if skewness < 2 else 0.8
        return "J-shape", confidence
    
    # L-shape characteristics (left side higher)
    if skewness < -0.8 and left_avg > right_avg * 2 and left_avg > middle_avg:
        confidence = 0.7 + 0.2 * (abs(skewness) / 2) if abs(skewness) < 2 else 0.8
        return "L-shape", confidence
    
    # Fallback cases
    if var_ratio > 2 and left_avg > right_avg:
        return "L-shape (weak)", 0.6
        
    if var_ratio < 0.5 and right_avg > left_avg:
        return "J-shape (weak)", 0.6
        
    if quad_coef > 0.2:
        return "U-shape (weak)", 0.5
        
    if quad_coef < -0.2:
        return "Inverted U-shape (weak)", 0.5
    
    return "Undetermined", 0.4


def plot_curve(x, y, title):
    """Plot a single curve with its title"""
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'o-', markersize=3)
    plt.title(title)
    plt.xlabel('Measurement Value')
    plt.ylabel('Defect Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_and_classify_curves():
    """Generate, analyze, and classify sample curves"""
    # Generate sample curves
    curves = generate_sample_curves(n_points=100, noise_level=0.03)
    
    # Create dataframe to store results
    results = []
    
    # Analyze each curve
    for curve_type, data in curves.items():
        x = data['measurement']
        y = data['defect_rate']
        
        # Compute metrics
        metrics = compute_metrics(x, y)
        
        # Classify curve shape
        predicted_shape, confidence = classify_curve_shape(metrics)
        
        # Store results
        result = {
            'Actual Type': curve_type,
            'Predicted Type': predicted_shape,
            'Confidence': f"{confidence:.2f}",
            **{k: f"{v:.2f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else v 
               for k, v in metrics.items()}
        }
        results.append(result)
        
        # Plot the curve
        plot_curve(x, y, f"Actual: {curve_type}, Predicted: {predicted_shape} (Conf: {confidence:.2f})")
    
    # Create DataFrame for better visualization
    results_df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    key_columns = ['Actual Type', 'Predicted Type', 'Confidence', 'quadratic_coefficient', 
                  'r_squared', 'skewness', 'kurtosis', 'left_avg', 'middle_avg', 'right_avg', 
                  'edge_to_center_ratio']
    other_columns = [col for col in results_df.columns if col not in key_columns]
    results_df = results_df[key_columns + other_columns]
    
    # Print results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nCurve Classification Results:")
    print(results_df)
    
    return results_df, curves


def classify_new_curve(measurements, defect_rates):
    """
    Classify a new defect rate curve
    
    Args:
        measurements: Array of measurement values
        defect_rates: Array of corresponding defect rates
        
    Returns:
        Predicted shape and confidence
    """
    metrics = compute_metrics(measurements, defect_rates)
    shape, confidence = classify_curve_shape(metrics)
    
    # Print key metrics
    print(f"Key Metrics:")
    print(f"- Skewness: {metrics['skewness']:.2f}")
    print(f"- Kurtosis: {metrics['kurtosis']:.2f}")
    print(f"- Quadratic Coefficient: {metrics['quadratic_coefficient']:.2f}")
    print(f"- R-squared: {metrics['r_squared']:.2f}")
    print(f"- Edge to Center Ratio: {metrics['edge_to_center_ratio']:.2f}")
    print(f"- Left/Right Averages: {metrics['left_avg']:.2f} / {metrics['right_avg']:.2f}")
    
    # Plot the curve
    plot_curve(measurements, defect_rates, f"Predicted: {shape} (Confidence: {confidence:.2f})")
    
    return shape, confidence, metrics


# Run the analysis
if __name__ == "__main__":
    results, curves = analyze_and_classify_curves()
    
    # Example: Classify a new curve
    print("\n\nExample: Classifying a new curve...")
    # Generate a new J-shaped curve with different parameters
    x_new = np.linspace(-5, 5, 100)
    y_new = 0.1 + 0.7 * np.exp(0.6 * (x_new - 1)) / (1 + np.exp(0.6 * (x_new - 1))) + 0.04 * np.random.randn(100)
    
    shape, confidence, _ = classify_new_curve(x_new, y_new)
    print(f"\nClassification result: {shape} with confidence {confidence:.2f}")

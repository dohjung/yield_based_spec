import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kurtosis, weibull_min
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import chi2

# 1. Generate sample curves
np.random.seed(42)
x = np.linspace(-3, 3, 300)
defect_u = 0.1 + 0.5 * (x**2) + np.random.normal(0, 0.05, size=x.size)
defect_inv_u = np.clip(0.6 - 0.5 * (x**2) + np.random.normal(0, 0.05, size=x.size), 0, None)
defect_l = np.where(x < -1, 0.6 + 0.5 * (-x - 1), 0.1 + np.random.normal(0, 0.02, size=x.size))
defect_j = np.where(x > 1, 0.6 + 0.5 * (x - 1), 0.1 + np.random.normal(0, 0.02, size=x.size))

curves_df = pd.DataFrame({
    'x': x,
    'U_shape': defect_u,
    'Inverted_U_shape': defect_inv_u,
    'L_shape': defect_l,
    'J_shape': defect_j
})

# 2. Plot curves
plt.figure(figsize=(10, 6))
plt.plot(x, defect_u, label='U-shaped')
plt.plot(x, defect_inv_u, label='Inverted U-shaped')
plt.plot(x, defect_l, label='L-shaped')
plt.plot(x, defect_j, label='J-shaped')
plt.xlabel('Measurement Value')
plt.ylabel('Defect Rate')
plt.title('Sample Defect Rate Curves')
plt.legend()
plt.grid(True)
plt.show()

# 3. Metric computation functions
def segmented_spearman(x, y):
    left_corr, _ = spearmanr(x[x < 0], y[x < 0])
    right_corr, _ = spearmanr(x[x >= 0], y[x >= 0])
    return left_corr, right_corr

def fit_weibull(y):
    params = weibull_min.fit(y, floc=0)
    return params[0]  # shape parameter

def var_to_mean(y):
    return np.var(y) / np.mean(y)

def threshold_exceedance(x, y):
    upper_limit = 2
    lower_limit = -2
    upper_exceed = np.mean(y[x > upper_limit])
    lower_exceed = np.mean(y[x < lower_limit])
    return lower_exceed, upper_exceed

def quadratic_regression(x, y):
    X_lin = x.reshape(-1, 1)
    X_quad = PolynomialFeatures(degree=2).fit_transform(X_lin)
    model_lin = LinearRegression().fit(X_lin, y)
    model_quad = LinearRegression().fit(X_quad, y)
    y_pred_lin = model_lin.predict(X_lin)
    y_pred_quad = model_quad.predict(X_quad)
    rss_lin = np.sum((y - y_pred_lin)**2)
    rss_quad = np.sum((y - y_pred_quad)**2)
    n = len(y)
    p_lin = 2
    p_quad = 3
    lr_stat = n * np.log(rss_lin / rss_quad)
    p_value = chi2.sf(lr_stat, df=p_quad - p_lin)
    quad_coef = model_quad.coef_[2]
    return quad_coef, p_value

# 4. Sequentially compute metrics for each curve
results = {}
for curve_name in ['U_shape', 'Inverted_U_shape', 'L_shape', 'J_shape']:
    y = curves_df[curve_name].values
    x_vals = curves_df['x'].values
    spearman_corr, _ = spearmanr(x_vals, y)
    seg_left, seg_right = segmented_spearman(x_vals, y)
    weibull_shape = fit_weibull(y)
    kurt = kurtosis(y)
    vtm = var_to_mean(y)
    lower_exceed, upper_exceed = threshold_exceedance(x_vals, y)
    quad_coef, p_val = quadratic_regression(x_vals, y)
    results[curve_name] = {
        'Spearman Correlation': spearman_corr,
        'Segmented Spearman Left': seg_left,
        'Segmented Spearman Right': seg_right,
        'Weibull Shape Parameter': weibull_shape,
        'Kurtosis': kurt,
        'Variance to Mean Ratio': vtm,
        'Threshold Exceedance Lower': lower_exceed,
        'Threshold Exceedance Upper': upper_exceed,
        'Quadratic Coefficient': quad_coef,
        'Quadratic Model p-value': p_val
    }

# 5. Display results
import pprint
pprint.pprint(results)


#====================================================================
def identify_defect_curve_shape(metrics):
    """
    Identifies the shape of defect rate curves based on statistical metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing the following metrics:
        - 'Spearman Correlation': Overall correlation between measurement and defect rate
        - 'Segmented Spearman Left': Correlation for values below the mean/median
        - 'Segmented Spearman Right': Correlation for values above the mean/median
        - 'Weibull Shape Parameter': Shape parameter from Weibull distribution fit
        - 'Kurtosis': Measure of the "tailedness" of the distribution
        - 'Variance to Mean Ratio': Dispersion relative to central tendency
        - 'Threshold Exceedance Lower': Average defect rate below lower threshold
        - 'Threshold Exceedance Upper': Average defect rate above upper threshold
        - 'Quadratic Coefficient': Coefficient of the quadratic term in regression
        - 'Quadratic Model p-value': Significance of the quadratic model
    
    Returns:
    --------
    str
        Identified shape: 'U-shaped', 'Inverted U-shaped', 'L-shaped', 'J-shaped', 
        'Possibly U-shaped', or 'Unclassified'
    dict
        Diagnostic information about which criteria were met
    """
    # Initialize results dictionary to track which criteria were met
    diagnostics = {
        'U_shape_criteria': False,
        'Inverted_U_criteria': False,
        'L_shape_criteria': False,
        'J_shape_criteria': False,
        'High_kurtosis': False,
        'High_vtm': False,
        'Significant_quadratic': False
    }
    
    # Extract metrics
    spearman = metrics['Spearman Correlation']
    seg_left = metrics['Segmented Spearman Left']
    seg_right = metrics['Segmented Spearman Right']
    weibull_shape = metrics['Weibull Shape Parameter']
    kurt = metrics['Kurtosis']
    vtm = metrics['Variance to Mean Ratio']
    lower_exceed = metrics['Threshold Exceedance Lower']
    upper_exceed = metrics['Threshold Exceedance Upper']
    quad_coef = metrics['Quadratic Coefficient']
    quad_p_val = metrics['Quadratic Model p-value']
    
    # Update diagnostics based on metrics
    diagnostics['High_kurtosis'] = kurt > 3
    diagnostics['High_vtm'] = vtm > 1.5
    diagnostics['Significant_quadratic'] = quad_p_val < 0.05
    
    # Check for various curve shapes
    
    # U-shaped criteria: quadratic coefficient positive and significant,
    # segmented Spearman correlations negative on both sides
    u_shape_criteria = (
        quad_coef > 0 and 
        quad_p_val < 0.05 and 
        seg_left < -0.3 and 
        seg_right > 0.3
    )
    diagnostics['U_shape_criteria'] = u_shape_criteria
    
    # Inverted U-shaped criteria: quadratic coefficient negative and significant,
    # segmented Spearman correlations positive on both sides
    inv_u_criteria = (
        quad_coef < 0 and 
        quad_p_val < 0.05 and 
        seg_left > 0.3 and 
        seg_right < -0.3
    )
    diagnostics['Inverted_U_criteria'] = inv_u_criteria
    
    # L-shaped criteria: strong negative correlation,
    # high threshold exceedance on lower side
    l_shape_criteria = (
        spearman < -0.5 and 
        lower_exceed > 0.3 and
        upper_exceed < 0.2
    )
    diagnostics['L_shape_criteria'] = l_shape_criteria
    
    # J-shaped criteria: strong positive correlation,
    # high threshold exceedance on upper side
    j_shape_criteria = (
        spearman > 0.5 and 
        upper_exceed > 0.3 and
        lower_exceed < 0.2
    )
    diagnostics['J_shape_criteria'] = j_shape_criteria
    
    # Determine shape based on criteria
    if u_shape_criteria:
        shape = "U-shaped"
    elif inv_u_criteria:
        shape = "Inverted U-shaped"
    elif l_shape_criteria:
        shape = "L-shaped"
    elif j_shape_criteria:
        shape = "J-shaped"
    # Additional checks if primary criteria not met
    elif kurt > 3 and abs(quad_coef) > 0.3:
        shape = "Possibly U-shaped (high kurtosis)"
    elif vtm > 2.0:
        shape = "Possibly non-linear (high variance)"
    else:
        shape = "Unclassified"
    
    return shape, diagnostics


def generate_shape_metrics_report(data_dict):
    """
    Processes a dictionary of measurement-defect data
    and returns a comprehensive report on curve shapes.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are dataset names and values are dictionaries of metrics
    
    Returns:
    --------
    dict
        Report containing identified shapes and diagnostics for each dataset
    """
    report = {}
    
    for dataset_name, metrics in data_dict.items():
        shape, diagnostics = identify_defect_curve_shape(metrics)
        report[dataset_name] = {
            'Identified_Shape': shape,
            'Diagnostics': diagnostics,
            'Key_Metrics': {
                'Quadratic_Coefficient': metrics['Quadratic Coefficient'],
                'Spearman_Correlation': metrics['Spearman Correlation'],
                'Segmented_Left_Right': [metrics['Segmented Spearman Left'], 
                                         metrics['Segmented Spearman Right']],
                'Exceedance_Lower_Upper': [metrics['Threshold Exceedance Lower'],
                                          metrics['Threshold Exceedance Upper']]
            }
        }
    
    return report


# Example usage with sample data
if __name__ == "__main__":
    # Sample metrics from different curve shapes
    sample_data = {
        'Dataset_1': {
            'Spearman Correlation': 0.05,
            'Segmented Spearman Left': -0.7,
            'Segmented Spearman Right': 0.8,
            'Weibull Shape Parameter': 1.8,
            'Kurtosis': 4.2,
            'Variance to Mean Ratio': 2.5,
            'Threshold Exceedance Lower': 0.4,
            'Threshold Exceedance Upper': 0.5,
            'Quadratic Coefficient': 0.8,
            'Quadratic Model p-value': 0.001
        },
        'Dataset_2': {
            'Spearman Correlation': -0.1,
            'Segmented Spearman Left': 0.6,
            'Segmented Spearman Right': -0.7,
            'Weibull Shape Parameter': 0.9,
            'Kurtosis': 2.1,
            'Variance to Mean Ratio': 1.2,
            'Threshold Exceedance Lower': 0.1,
            'Threshold Exceedance Upper': 0.1,
            'Quadratic Coefficient': -0.6,
            'Quadratic Model p-value': 0.003
        },
        'Dataset_3': {
            'Spearman Correlation': -0.8,
            'Segmented Spearman Left': -0.9,
            'Segmented Spearman Right': -0.2,
            'Weibull Shape Parameter': 1.5,
            'Kurtosis': 1.2,
            'Variance to Mean Ratio': 0.8,
            'Threshold Exceedance Lower': 0.7,
            'Threshold Exceedance Upper': 0.1,
            'Quadratic Coefficient': 0.2,
            'Quadratic Model p-value': 0.3
        },
        'Dataset_4': {
            'Spearman Correlation': 0.85,
            'Segmented Spearman Left': 0.2,
            'Segmented Spearman Right': 0.9,
            'Weibull Shape Parameter': 2.2,
            'Kurtosis': 1.8,
            'Variance to Mean Ratio': 0.9,
            'Threshold Exceedance Lower': 0.1,
            'Threshold Exceedance Upper': 0.8,
            'Quadratic Coefficient': 0.1,
            'Quadratic Model p-value': 0.4
        }
    }
    
    # Generate and print the report
    report = generate_shape_metrics_report(sample_data)
    
    print("\nDefect Curve Shape Analysis Report")
    print("=" * 40)
    for dataset, results in report.items():
        print(f"\nDataset: {dataset}")
        print(f"Identified Shape: {results['Identified_Shape']}")
        print("Key Metrics:")
        for metric, value in results['Key_Metrics'].items():
            print(f"  - {metric}: {value}")
    
    print("\nDetailed Diagnostics:")
    print("=" * 40)
    for dataset, results in report.items():
        print(f"\n{dataset} - {results['Identified_Shape']}:")
        for criterion, met in results['Diagnostics'].items():
            print(f"  - {criterion}: {'✓' if met else '✗'}")

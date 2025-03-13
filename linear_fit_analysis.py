"""
Linear Fitting Analysis for Scope Height vs Distance

This script analyzes how well a linear model fits the relationship 
between scope height and target distance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from simple_calibration import create_calibration_table
import pandas as pd


def linear_regression(x, y):
    """
    Perform linear regression and return slope, intercept, and r-value

    Args:
        x: x values (distances)
        y: y values (scope heights)

    Returns:
        tuple: (slope, intercept, r_value, p_value, std_err)
    """
    return stats.linregress(x, y)


def calculate_errors(x, y, slope, intercept):
    """
    Calculate various error metrics between actual values and linear fit

    Args:
        x: x values (distances)
        y: actual y values (scope heights)
        slope: slope of linear fit
        intercept: intercept of linear fit

    Returns:
        dict: Dictionary of error metrics
    """
    # Predicted values from linear model
    y_pred = slope * x + intercept

    # Absolute errors
    abs_errors = np.abs(y - y_pred)

    # Relative errors (percentage)
    rel_errors = abs_errors / y * 100

    # Mean squared error
    mse = np.mean((y - y_pred) ** 2)

    # Root mean squared error
    rmse = np.sqrt(mse)

    # Mean absolute error
    mae = np.mean(abs_errors)

    # Maximum error
    max_error = np.max(abs_errors)
    max_error_index = np.argmax(abs_errors)
    max_error_point = (x[max_error_index], y[max_error_index])

    # R-squared (coefficient of determination)
    # R^2 = 1 - SSres/SStot
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        'absolute_errors': abs_errors,
        'relative_errors': rel_errors,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'max_error_point': max_error_point,
        'r_squared': r_squared,
        'predicted_values': y_pred
    }


def plot_results(x, y, slope, intercept, errors):
    """
    Plot the results of linear fitting and errors

    Args:
        x: x values (distances)
        y: actual y values (scope heights)
        slope: slope of linear fit
        intercept: intercept of linear fit
        errors: dictionary of error metrics
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Actual data vs linear fit
    ax1.scatter(x, y, color='blue', s=30, alpha=0.7, label='Actual Data')

    # Linear fit line
    y_pred = errors['predicted_values']
    ax1.plot(x, y_pred, 'r-', linewidth=2,
             label=f'Linear Fit: y = {slope:.6f}x + {intercept:.6f}')

    # Add title and labels
    ax1.set_title('Scope Height vs Distance Linear Fit', fontsize=14)
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Scope Height (m)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Equation and R-squared text
    equation_text = f"Linear Model: y = {slope:.6f}x + {intercept:.6f}\n"
    equation_text += f"R² = {errors['r_squared']:.6f}\n"
    equation_text += f"RMSE = {errors['rmse']:.6f}\n"
    equation_text += f"Maximum Error = {errors['max_error']:.6f} at distance {errors['max_error_point'][0]:.1f}m"

    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.05, equation_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)

    # Plot 2: Errors
    ax2.plot(x, errors['absolute_errors'] * 1000, 'g-',
             linewidth=2, label='Absolute Error (mm)')

    # Add secondary y-axis for relative errors
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, errors['relative_errors'], 'm--',
                  linewidth=2, label='Relative Error (%)')
    ax2_twin.set_ylabel('Relative Error (%)', color='m')

    # Highlight maximum error point
    max_x, max_y = errors['max_error_point']
    max_error = errors['max_error'] * 1000  # convert to mm
    ax2.scatter([max_x], [max_error], color='red', s=80,
                zorder=5, label=f'Max Error: {max_error:.2f}mm')

    # Add title and labels
    ax2.set_title('Error Analysis', fontsize=14)
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Absolute Error (mm)')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('linear_fit_analysis.png', dpi=300)
    plt.show()


def print_detailed_errors(x, y, slope, intercept, errors):
    """
    Print detailed error analysis

    Args:
        x: x values (distances)
        y: actual y values (scope heights)
        slope: slope of linear fit
        intercept: intercept of linear fit
        errors: dictionary of error metrics
    """
    # Create a DataFrame for nice output
    df = pd.DataFrame({
        'Distance (m)': x,
        'Actual Scope Height (m)': y,
        'Predicted Scope Height (m)': errors['predicted_values'],
        # Convert to mm
        'Absolute Error (mm)': errors['absolute_errors'] * 1000,
        'Relative Error (%)': errors['relative_errors']
    })

    # Print summary statistics
    print("\n=== Linear Fit Analysis ===")
    print(f"Linear Model: y = {slope:.6f}x + {intercept:.6f}")
    print(f"R² (Coefficient of Determination): {errors['r_squared']:.6f}")
    print(
        f"Root Mean Square Error (RMSE): {errors['rmse']:.6f} m ({errors['rmse']*1000:.2f} mm)")
    print(
        f"Mean Absolute Error (MAE): {errors['mae']:.6f} m ({errors['mae']*1000:.2f} mm)")
    print(
        f"Maximum Error: {errors['max_error']:.6f} m ({errors['max_error']*1000:.2f} mm) at distance {errors['max_error_point'][0]:.1f}m")

    # Print average error by distance ranges
    print("\n=== Error by Distance Range ===")

    ranges = [(20, 40), (41, 60), (61, 80), (81, 100)]
    for start, end in ranges:
        # Filter by range
        mask = (x >= start) & (x <= end)
        if np.any(mask):
            avg_abs_error = np.mean(
                errors['absolute_errors'][mask]) * 1000  # Convert to mm
            avg_rel_error = np.mean(errors['relative_errors'][mask])
            print(
                f"Range {start}-{end}m: Avg Abs Error = {avg_abs_error:.2f} mm, Avg Rel Error = {avg_rel_error:.2f}%")

    # Print detailed table (sample rows for brevity)
    print("\n=== Detailed Error Table (Sample) ===")
    sample_indices = list(range(0, len(x), 10))  # Every 10th point
    print(df.iloc[sample_indices].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))

    # Save full table to CSV
    df.to_csv('linear_fit_analysis.csv', index=False, float_format='%.6f')
    print("\nFull analysis saved to linear_fit_analysis.csv")


def analyze_linear_fit():
    """
    Analyze how well a linear model fits the relationship between scope height and distance
    """
    # Generate calibration data with 1m step
    calibration_data = create_calibration_table((20, 100), 1)

    # Extract distances and scope heights
    distances = np.array([data['distance'] for data in calibration_data])
    scope_heights = np.array([data['ym'] for data in calibration_data])

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linear_regression(
        distances, scope_heights)

    # Calculate errors
    errors = calculate_errors(distances, scope_heights, slope, intercept)

    # Print detailed error analysis
    print_detailed_errors(distances, scope_heights, slope, intercept, errors)

    # Plot results
    plot_results(distances, scope_heights, slope, intercept, errors)

    return slope, intercept, errors


if __name__ == "__main__":
    slope, intercept, errors = analyze_linear_fit()

    # Print simplified linear equation for practical use
    print("\n=== Simplified Linear Equation for Practical Use ===")
    print(f"Scope Height (m) = {slope:.6f} × Distance (m) + {intercept:.6f}")

    # Example usage
    test_distance = 65  # meters
    predicted_height = slope * test_distance + intercept
    print(
        f"\nExample: At {test_distance}m distance, estimated scope height = {predicted_height:.4f}m")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator

# Function to read data from each worksheet
def read_yield_data(file_path):
    # Read all sheets into a dictionary of dataframes
    sheets = pd.read_excel(file_path, sheet_name=None)
    yield_data = {}
    
    for sheet_name, df in sheets.items():
        # Ensure the DataFrame has correct structure
        df = df[['Years left(precise)', 'Yield to Maturity']]  # Adjust column names as needed
        yield_data[sheet_name] = df.dropna()  # Drop rows with missing values
        
    return yield_data

"""# Function to interpolate missing maturities and align all data
def interpolate_yields(yield_data, maturities_to_interpolate):
    interpolated_data = {}
    
    for date, df in yield_data.items():
        x = df['Years left(precise)']
        y = df['Yield to Maturity']
        
        # Cubic spline interpolation
        cs = CubicSpline(x, y)
        interpolated_yields = cs(maturities_to_interpolate)
        
        interpolated_data[date] = interpolated_yields
        
    return interpolated_data"""
def interpolate_yields(yield_data, maturities_to_interpolate):
    interpolated_data = {}
    
    for date, df in yield_data.items():
        x = df['Years left(precise)']
        y = df['Yield to Maturity']
        
        # Check if x is strictly increasing
        if np.all(np.diff(x) > 0):
            # PCHIP interpolation
            pchip = PchipInterpolator(x, y)
            interpolated_yields = pchip(maturities_to_interpolate)
            interpolated_data[date] = interpolated_yields
        else:
            print(f"Skipping {date}: Maturities are not strictly increasing.")
        
    return interpolated_data

# Function to plot the yield curves
def plot_yield_curves(interpolated_data, maturities_to_interpolate):
    plt.figure(figsize=(10, 6))
    
    for date, yields in interpolated_data.items():
        # Convert decimal YTM values to percentages
        yields_percent = yields * 100  # Multiply by 100 to convert to percentages
        plt.plot(maturities_to_interpolate, yields_percent, label=date)
    
    # Set Y-axis limits for a broader scale
    plt.ylim(0, 5)  # Set Y-axis scale from 0% to 5%
    
    plt.title('Yield Curves for Different Dates')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Yield to Maturity (%)')  # Update Y-axis label
    plt.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Main script
file_path = 'Test.xlsx'  # Replace with your file path
maturities_to_interpolate = np.linspace(0, 5, 50)  # 0 to 5 years, 50 points for smooth curves

# Step 1: Read the data
yield_data = read_yield_data(file_path)

# Step 2: Interpolate yields
interpolated_data = interpolate_yields(yield_data, maturities_to_interpolate)

# Step 3: Plot the yield curves
plot_yield_curves(interpolated_data, maturities_to_interpolate)

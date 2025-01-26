import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def interpolate_yields(yield_data, maturities_to_interpolate):
    interpolated_data = {}
    # Interpolate yields for each date
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
            # Print a warning message if x is not strictly increasing
            print(f"Skipping {date}: Maturities are not strictly increasing.")
        
    return interpolated_data

# Function to plot the yield curves
def plot_yield_curves(interpolated_data, maturities_to_interpolate):
    plt.figure(figsize=(10, 6))
    
    # Plot interpolated yield curves for each date
    for date, yields in interpolated_data.items():
        # Convert decimal YTM values to percentages
        yields_percent = yields * 100  # Multiply by 100 to convert to percentages
        plt.plot(maturities_to_interpolate, yields_percent, label=date)
    
    # Set Y-axis limits for a broader scale
    plt.ylim(0, 5)  # Set Y-axis scale from 0% to 5%
    
    # Add labels, title, legend, grid, and show the plot
    plt.title('Yield Curves for Different Dates')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Yield to Maturity (%)')  # Update Y-axis label
    plt.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = 'Test.xlsx'  
    maturities_to_interpolate = np.linspace(0, 5, 50)
    yield_data = read_yield_data(file_path)
    interpolated_data = interpolate_yields(yield_data, maturities_to_interpolate)
    plot_yield_curves(interpolated_data, maturities_to_interpolate)



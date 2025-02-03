import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve

# ---------------------------
# Helper functions for cleaning
# ---------------------------
def clean_percentage(x):
    if isinstance(x, str):
        return float(x.replace('%', '').strip())
    return x

def clean_yield_dataframe(df):
    """
    Clean a dataframe to ensure 'Years left(precise)' and 'Yield to Maturity'
    are numeric and properly formatted.
    """
    df = df[['Years left(precise)', 'Yield to Maturity']].dropna()
    df['Years left(precise)'] = pd.to_numeric(df['Years left(precise)'], errors='coerce')
    df['Yield to Maturity'] = pd.to_numeric(df['Yield to Maturity'].astype(str).str.replace('%', ''), errors='coerce') / 100.0
    return df

def select_closest_row(df, target, col):
    """
    Select the row from df for which the absolute difference between the value in 'col'
    and the 'target' is minimized.
    """
    diff = (df[col] - target).abs()
    return df.loc[diff.idxmin()]

# ---------------------------
# PART 4(a): Interpolated Yield Curve
# ---------------------------
def read_yield_data(file_path):
    """
    Read an Excel file containing bond yield data.
    Returns a dictionary of dataframes, one for each sheet.
    Each dataframe should have columns "Years left(precise)" and "Yield to Maturity"
    """
    sheets = pd.read_excel(file_path, sheet_name=None)
    # For yield interpolation we need: "Years left(precise)" and "Yield to Maturity"
    data = {}
    # Clean and convert columns to numeric
    for name, df in sheets.items():
        # Check required columns exist
        if 'Years left(precise)' in df.columns and 'Yield to Maturity' in df.columns:
            # Clean the yield column (remove '%' if present) and convert to float
            df = df.dropna(subset=['Years left(precise)', 'Yield to Maturity'])
            df['Yield to Maturity'] = df['Yield to Maturity'].apply(clean_percentage)
            # Convert yields to decimals
            df['Yield to Maturity'] = df['Yield to Maturity'] / 100.0
            data[name] = df
    return data

def plot_yield_curves(file_path):
    """
    Given an Excel file containing bond yield data, plot the interpolated yield curves.
    """
    data = read_yield_data(file_path)
    maturities = np.linspace(0, 5, 50)
    plt.figure(figsize=(8, 5))
    
    # Plot each yield curve
    for name, df in data.items():
        # Sort by maturity
        x = df['Years left(precise)'].values
        y = df['Yield to Maturity'].values
        idx = np.argsort(x)
        x, y = x[idx], y[idx]

        # Interpolate with PCHIP (shape-preserving)
        interp = PchipInterpolator(x, y)
        # Evaluate the interpolated curve at the desired maturities
        # Multiply by 10000 to get percentage points
        y_interp = interp(maturities) * 10000  

        # Plot the interpolated curve
        plt.plot(maturities, y_interp, label=name)

    plt.xlabel("Maturity (Years)", color='black')
    plt.ylabel("Yield to Maturity (%)", color='black')
    plt.title("Interpolated 5-Year Yield Curves", color='black')
    plt.legend()
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    #set y axis to 2% to 4.5%
    plt.ylim(2, 4.5)
    #have legend outside of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()


# ---------------------------
# PART 4(b): Bootstrapped Spot Curve
# ---------------------------
def select_5_bonds(df):
    """
    From a dataframe containing bond data (columns: Price, Coupon, Years left(precise)),
    select 5 bonds whose maturities are closest to 1, 2, 3, 4, and 5 years.
    """
    # Drop rows with missing values
    df = df[['Price', 'Coupon', 'Years left(precise)']].dropna().copy()
    # Ensure numeric values for "Years left(precise)"
    df["Years left(precise)"] = pd.to_numeric(df["Years left(precise)"], errors='coerce')
    selected = []
    targets = [1, 2, 3, 4, 5]
    # Find the bond with maturity closest to each target
    for t in targets:
        row = select_closest_row(df, t, "Years left(precise)")
        coupon = clean_percentage(row['Coupon'])
        selected.append((row['Price'], coupon, row["Years left(precise)"]))
    return selected

def bootstrap_spot_curve(bonds):
    """
    Given a list of bonds (each as (Price, Coupon, Maturity)), bootstrap the spot rates.
    Assume semiannual compounding on a $100 face value.
    """
    spot_rates = []
    # Sort bonds by maturity (smallest first)
    bonds = sorted(bonds, key=lambda x: x[2])
    # Bootstrap spot rates 
    for i, (price, coupon, maturity) in enumerate(bonds):
        # Calculate semi-annual coupon payment
        coupon_payment = coupon * 100 / 2.0 
        # Calculate spot rate
        if i == 0:
            # First bond: use coupon payment and face value
            r = 2 * (((100 + coupon_payment) / price)**(1/(2 * maturity)) - 1)
        else:
            def eq(r):
                # Discount previous coupon payments using known spot rates
                cf_sum = 0
                # Sum of discounted cash flows from previous bonds
                for j in range(i):
                    cf_sum += coupon_payment / ((1 + spot_rates[j]/2) ** (2*(j+1)))
                # Final cash flow includes coupon + face value
                temp = price - (cf_sum + (coupon_payment + 100) / ((1 + r/2) ** (2*maturity)))
                return temp
            # Solve for spot rate using the bond price and cash flows from previous bonds 
            r = fsolve(eq, 0.05)[0]
        spot_rates.append(r)
    return np.array(spot_rates)

def plot_spot_curve(file_path):
    """
    Given an Excel file containing bond data, plot the bootstrapped spot curves.
    """
    sheets = pd.read_excel(file_path, sheet_name=None)
    plt.figure(figsize=(8, 5))
    # Plot each spot curve
    for name, df in sheets.items():
        # Select 5 bonds with maturities closest to 1, 2, 3, 4, 5 years
        bonds = select_5_bonds(df)
        # Bootstrap spot curve
        spot = bootstrap_spot_curve(bonds)
        maturities = [1, 2, 3, 4, 5]
        plt.plot(maturities, spot*100, marker='o', label=name)
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Spot Rate (%)")
    plt.title("Bootstrapped 5-Year Spot Curves")
    #have legend outside of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------
# PART 4(c): 1-Year Forward Curve
# ---------------------------
def calculate_forward_rates(spot_rates):
    """
    Given a list of spot rates, calculate the 1-year forward rates.
    """
    forward = []
    S1 = spot_rates[0]
    # Calculate forward rates for 2 to 5 years
    for j in range(1, 5):
        Sj = spot_rates[j]
        # Calculate forward rate using spot rates
        f = ((1+Sj)**(2*(j+1))/(1+S1)**2)**(1/(2*j)) - 1
        forward.append(f)
    return np.array(forward)

def plot_forward_curve(file_path):
    """
    Given an Excel file containing bond data, plot the 1-year forward curves.
    """
    sheets = pd.read_excel(file_path, sheet_name=None)
    plt.figure(figsize=(8, 5))
    for name, df in sheets.items():
        bonds = select_5_bonds(df)
        spot = bootstrap_spot_curve(bonds)
        forward = calculate_forward_rates(spot)
        maturities = [2, 3, 4, 5]
        plt.plot(maturities, forward*100, marker='o', label=name)
    plt.xlabel("Maturity (End Year of Forward Period)")
    plt.ylabel("1-Year Forward Rate (%)")
    plt.title("1-Year Forward Curves")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------
# PART 5: Covariance Matrices
# ---------------------------
def ytm_data(file_path):
    """
    Extract fixed maturity yields from an Excel file.
    """
    sheets = pd.read_excel(file_path, sheet_name=None)
    maturities = [1, 2, 3, 4, 5]
    data = {}
    for name, df in sheets.items():
        df = clean_yield_dataframe(df)
        data[name] = [
            select_closest_row(df, m, 'Years left(precise)')['Yield to Maturity']
            for m in maturities
        ]
    return pd.DataFrame.from_dict(data, orient='index', columns=['1yr', '2yr', '3yr', '4yr', '5yr'])

def calc_log_returns(df):
    """
    Calculate log returns of a dataframe.
    """
    return np.log(df / df.shift(1)).dropna()

def calc_cov_matrix(log_returns):
    """
    Compute the covariance matrix of log returns.
    """
    return np.cov(log_returns.T)

def forward_data(spot_data):
    """
    Extract forward rates from spot rates.
    """
    fr_data = {date: calculate_forward_rates(row.values) for date, row in spot_data.iterrows()}
    return pd.DataFrame.from_dict(fr_data, orient='index', columns=['1yr-1yr', '1yr-2yr', '1yr-3yr', '1yr-4yr'])

def spot_rates(file_path):
    sheets = pd.read_excel(file_path, sheet_name=None)
    req = {'Price', 'Coupon', 'Years left(precise)'}
    data = {}
    for name, df in sheets.items():
        if req.issubset(df.columns):
            bonds = select_5_bonds(df)
            data[name] = bootstrap_spot_curve(bonds)
    return pd.DataFrame.from_dict(data, orient='index', columns=['1yr', '2yr', '3yr', '4yr', '5yr'])


# ---------------------------
# PART 6: Eigenvectors/Eigenvalues
# ---------------------------
def calc_eigen_val_vector(covariance_matrix):
    """
    Compute eigenvalues and eigenvectors.
    """
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)
    return eigenvalues, eigenvectors

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    file_path = "data.xlsx" 
    print("Plotting Yield Curves ...")
    plot_yield_curves(file_path)
    print("Plotting Spot Curves ...")
    plot_spot_curve(file_path)
    print("Plotting Forward Curves ...")
    plot_forward_curve(file_path)
    yield_data = ytm_data(file_path)
    yield_data_log_ret = calc_log_returns(yield_data)
    yield_cov_matrix = calc_cov_matrix(yield_data_log_ret)
    spot_data = spot_rates(file_path)
    forward_rates_data = forward_data(spot_data)
    forward_log_returns_data = calc_log_returns(forward_rates_data)
    forward_cov_matrix = calc_cov_matrix(forward_log_returns_data)
    print("\nCovariance Matrix of Daily Log Returns:\n", yield_cov_matrix)
    print("\nCovariance Matrix of Forward Rate Log Returns:\n", forward_cov_matrix)
    print("Eigvenvalues/Eigenvectors for yield rates covariance matrix: \n")
    yield_eigen_val_vec = calc_eigen_val_vector(yield_cov_matrix)
    print("Eigvenvalues/Eigenvectors for forward rates covariance matrix: \n")
    forward_eigen_val_vec = calc_eigen_val_vector(forward_cov_matrix)
    


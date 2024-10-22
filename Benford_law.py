import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare

def extract_first_digits(data):
    """
    Extracts the first significant digit from an array of positive integers.
    """
    data = np.abs(data)  # Ensure all values are positive
    first_digits = np.array([int(str(num)[0]) for num in data if num > 0])
    return first_digits

def benford_distribution():
    """
    Calculates the expected Benford distribution (log10(1 + 1/d)).
    """
    return np.log10(1 + 1 / np.arange(1, 10))

def empirical_distribution(first_digits):
    """
    Calculates the empirical distribution of the first digits.
    Returns Array of probabilities for digits 1-9.
    """
    counts = np.array([np.sum(first_digits == d) for d in range(1, 10)])
    return counts / counts.sum()

def benford_law(data):
    """
    Main function to calculate and plot the Benford and empirical distributions,
    and compute the p-value using a chi-square goodness-of-fit test.
    """
    # Extract first digits from the data
    first_digits = extract_first_digits(data)

    # Calculate Benford and empirical distributions
    benford_probs = benford_distribution()
    empirical_probs = empirical_distribution(first_digits)

    # Perform Chi-Square test for goodness-of-fit
    observed = np.array([np.sum(first_digits == d) for d in range(1, 10)])
    expected = benford_probs * observed.sum()
    chi2_stat, p_value = chisquare(observed, f_exp=expected)

    # Print results
    print(f"Chi-Square Statistic: {chi2_stat:.4f}")
    print(f"P-Value: {p_value:.4f}")

    # Dynamic comment based on p-value
    if p_value < 0.05:
        print("Anomaly Found")
    else:
        print("No Anomaly Found")

    # Plot the distributions
    plot_distributions(benford_probs, empirical_probs)

def plot_distributions(benford_probs, empirical_probs):
    """
    Plots the Benford distribution vs the empirical distribution.
    """
    digits = np.arange(1, 10)

    plt.figure(figsize=(10, 6))
    
    # Plot Benford Distribution as a bar chart
    plt.bar(digits - 0.2, benford_probs, width=0.4, 
            label='Benford Distribution', align='center', color='lightblue')
    
    # Plot Empirical Distribution as a line chart, aligned with bar centers
    plt.plot(digits, empirical_probs, marker='o', label='Empirical Distribution', 
             color='orange', linewidth=2)

    plt.xlabel('First Digit', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title('Benford vs Empirical Distribution', fontsize=16)
    plt.xticks(digits)
    plt.legend()
    plt.grid()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Load the dataset
    file_path = "/Users/mdshamiulislam/Downloads/US_County_Level_Presidential_Results_08-16.csv"
    df = pd.read_csv(file_path)

    # Prepare the data by selecting relevent columns and flattening the values
    data = df[['total_2016', 'dem_2016',	'gop_2016',	'oth_2016']].astype(int).values.flatten()

    # Call the benford_law function on the prepared data
    benford_law(data)
    

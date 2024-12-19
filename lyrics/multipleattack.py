import pandas as pd
import numpy as np

# File paths
files = [
    "results-gpt/gpt_run_gpt-3.5-turbo-0125.csv",
    "results-gpt/gpt_run_gpt-4o-mini.csv",
    "results-gpt/gpt_run_gpt-4o.csv"
]

# Function to calculate mean retries
def calculate_mean_retries(file_path):
    df = pd.read_csv(file_path)
    
    # Replace -1 with 4 in retries columns
    df['paraphrase_retries'] = df['paraphrase_retries'].replace(-1, 5)
    df['reverse_retries'] = df['reverse_retries'].replace(-1, 5)
    
    # Calculate mean
    mean_paraphrase = df['paraphrase_retries'].mean()
    mean_reverse = df['reverse_retries'].mean()
    
    return mean_paraphrase, mean_reverse

# Process all files
for file in files:
    try:
        mean_paraphrase, mean_reverse = calculate_mean_retries(file)
        print(f"File: {file}")
        print(f"Mean Paraphrase Retries: {mean_paraphrase:.2f}")
        print(f"Mean Reverse Retries: {mean_reverse:.2f}\n")
    except Exception as e:
        print(f"Error processing file {file}: {e}")


#for each file, write teh code that calculate the Correlation between Popularity and lcs and levenshtein distance
def calculate_correlation(file_path):
    df = pd.read_csv(file_path)
    # Calculate correlation
    corr_lcs_p = df['popularity'].corr(df['lcs_length'])
    corr_lev_p = df['popularity'].corr(df['levenshtein_distance'])
    corr_lcs_r = df['popularity'].corr(df['reverse_lcs_length'])
    corr_lev_r = df['popularity'].corr(df['reverse_levenshtein_distance'])
    
    return corr_lcs_p, corr_lev_p, corr_lcs_r, corr_lev_r
for file in files:
    try:
        corr_lcs_p, corr_lev_p, corr_lcs_r, corr_lev_r = calculate_correlation(file)
        print(f"File: {file}")
        print(f"Paraphrase Prompting - Correlation between Popularity and Normalized LCS Length: {corr_lcs_p:.4f}")
        print(f"Paraphrase Prompting - Correlation between Popularity and Normalized Levenshtein Distance: {corr_lev_p:.4f}")
        print(f"Reverse Prompting - Correlation between Popularity and Normalized LCS Length: {corr_lcs_r:.4f}")
        print(f"Reverse Prompting - Correlation between Popularity and Normalized Levenshtein Distance: {corr_lev_r:.4f}\n")
    except Exception as e:
        print(f"Error processing file {file}: {e}")


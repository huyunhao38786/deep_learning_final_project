import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the updated dataset from the specified path
dataset_path = 'dataset/music_dataset_with_strategies.csv'

# Load the dataset
data = pd.read_csv(dataset_path)

# Ensure that all necessary columns are present
required_columns = [
    'lyrics',
    'popularity',
    'lcs_paraphrase',
    'levenshtein_paraphrase',
    'lcs_reverse',
    'levenshtein_reverse'
]
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"The dataset does not contain a '{col}' column.")

# Compute the length of the original lyrics
data['original_lyrics_length'] = data['lyrics'].apply(len)

# Avoid division by zero
data = data[data['original_lyrics_length'] > 0]

# For Paraphrase Prompting
data['normalized_lcs_paraphrase'] = data['lcs_paraphrase'] / data['original_lyrics_length']
data['normalized_levenshtein_paraphrase'] = data['levenshtein_paraphrase'] / data['original_lyrics_length']

# For Reverse Prompting
data['normalized_lcs_reverse'] = data['lcs_reverse'] / data['original_lyrics_length']
data['normalized_levenshtein_reverse'] = data['levenshtein_reverse'] / data['original_lyrics_length']

# Plot settings
sns.set(style='whitegrid')

# Plot for Paraphrase Prompting - Normalized LCS Length
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_lcs_paraphrase', alpha=0.6)
plt.title('Normalized LCS Length vs. Popularity Score (Paraphrase Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized LCS Length')
plt.savefig('normalized_lcs_paraphrase.png')
plt.show()

# Plot for Reverse Prompting - Normalized LCS Length
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_lcs_reverse', alpha=0.6)
plt.title('Normalized LCS Length vs. Popularity Score (Reverse Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized LCS Length')
plt.savefig('normalized_lcs_reverse.png')
plt.show()

# Plot for Paraphrase Prompting - Normalized Levenshtein Distance
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_levenshtein_paraphrase', alpha=0.6)
plt.title('Normalized Levenshtein Distance vs. Popularity Score (Paraphrase Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized Levenshtein Distance')
plt.savefig('normalized_levenshtein_paraphrase.png')
plt.show()

# Plot for Reverse Prompting - Normalized Levenshtein Distance
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_levenshtein_reverse', alpha=0.6)
plt.title('Normalized Levenshtein Distance vs. Popularity Score (Reverse Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized Levenshtein Distance')
plt.savefig('normalized_levenshtein_reverse.png')
plt.show()

# Correlation for Paraphrase Prompting
corr_lcs_p = data['popularity'].corr(data['normalized_lcs_paraphrase'])
corr_lev_p = data['popularity'].corr(data['normalized_levenshtein_paraphrase'])
print(f"Paraphrase Prompting - Correlation between Popularity and Normalized LCS Length: {corr_lcs_p:.4f}")
print(f"Paraphrase Prompting - Correlation between Popularity and Normalized Levenshtein Distance: {corr_lev_p:.4f}")

# Correlation for Reverse Prompting
corr_lcs_r = data['popularity'].corr(data['normalized_lcs_reverse'])
corr_lev_r = data['popularity'].corr(data['normalized_levenshtein_reverse'])
print(f"Reverse Prompting - Correlation between Popularity and Normalized LCS Length: {corr_lcs_r:.4f}")
print(f"Reverse Prompting - Correlation between Popularity and Normalized Levenshtein Distance: {corr_lev_r:.4f}")

# Normalized LCS Length
mean_lcs_p = data['normalized_lcs_paraphrase'].mean()
median_lcs_p = data['normalized_lcs_paraphrase'].median()
std_lcs_p = data['normalized_lcs_paraphrase'].std()

# Normalized Levenshtein Distance
mean_lev_p = data['normalized_levenshtein_paraphrase'].mean()
median_lev_p = data['normalized_levenshtein_paraphrase'].median()
std_lev_p = data['normalized_levenshtein_paraphrase'].std()

print("Paraphrase Prompting Statistics:")
print(f"Normalized LCS Length - Mean: {mean_lcs_p:.4f}, Median: {median_lcs_p:.4f}, Standard Deviation: {std_lcs_p:.4f}")
print(f"Normalized Levenshtein Distance - Mean: {mean_lev_p:.4f}, Median: {median_lev_p:.4f}, Standard Deviation: {std_lev_p:.4f}")


# Normalized LCS Length
mean_lcs_r = data['normalized_lcs_reverse'].mean()
median_lcs_r = data['normalized_lcs_reverse'].median()
std_lcs_r = data['normalized_lcs_reverse'].std()

# Normalized Levenshtein Distance
mean_lev_r = data['normalized_levenshtein_reverse'].mean()
median_lev_r = data['normalized_levenshtein_reverse'].median()
std_lev_r = data['normalized_levenshtein_reverse'].std()

print("\nReverse Prompting Statistics:")
print(f"Normalized LCS Length - Mean: {mean_lcs_r:.4f}, Median: {median_lcs_r:.4f}, Standard Deviation: {std_lcs_r:.4f}")
print(f"Normalized Levenshtein Distance - Mean: {mean_lev_r:.4f}, Median: {median_lev_r:.4f}, Standard Deviation: {std_lev_r:.4f}")

# Create a DataFrame to summarize the statistics
statistics = pd.DataFrame({
    'Strategy': ['Paraphrase Prompting', 'Reverse Prompting'],
    'Mean Normalized LCS Length': [mean_lcs_p, mean_lcs_r],
    'Median Normalized LCS Length': [median_lcs_p, median_lcs_r],
    'Std Normalized LCS Length': [std_lcs_p, std_lcs_r],
    'Mean Normalized Levenshtein Distance': [mean_lev_p, mean_lev_r],
    'Median Normalized Levenshtein Distance': [median_lev_p, median_lev_r],
    'Std Normalized Levenshtein Distance': [std_lev_p, std_lev_r],
})

print("\nSummary Statistics:")
print(statistics)


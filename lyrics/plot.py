import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

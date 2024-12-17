import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = './dataset/13b'
os.makedirs(DATASET_DIR, exist_ok=True)

# Combine the checkpoint files
file_list = [f"checkpoint_results_{i}.csv" for i in range(1,5)]
dfs = [pd.read_csv(os.path.join(DATASET_DIR, fname), index_col=0) for fname in file_list]
data = pd.concat(dfs, ignore_index=True)

# Ensure required columns exist
required_cols = ['lyrics', 'lcs_paraphrase', 'levenshtein_paraphrase', 'lcs_reverse', 'levenshtein_reverse', 'popularity']
for col in required_cols:
    if col not in data.columns:
        raise KeyError(f"Column '{col}' is missing from the combined data.")

# Compute original lyrics length
data['original_lyrics_length'] = data['lyrics'].apply(len)

# Drop rows with zero-length lyrics to avoid division by zero
data = data[data['original_lyrics_length'] > 0]

# Compute normalized values
data['normalized_lcs_paraphrase'] = data['lcs_paraphrase'] / data['original_lyrics_length']
data['normalized_levenshtein_paraphrase'] = data['levenshtein_paraphrase'] / data['original_lyrics_length']

data['normalized_lcs_reverse'] = data['lcs_reverse'] / data['original_lyrics_length']
data['normalized_levenshtein_reverse'] = data['levenshtein_reverse'] / data['original_lyrics_length']

# Save the final dataset
output_path = os.path.join(DATASET_DIR, 'music_dataset_with_strategies_13b.csv')
data.to_csv(output_path, index=False)
print(f"Updated dataset saved to {output_path}")

# Create and save plots
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_lcs_paraphrase', alpha=0.6)
plt.title('Normalized LCS Length vs. Popularity Score (Paraphrase Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized LCS Length')
plt.savefig(os.path.join(DATASET_DIR, 'normalized_lcs_paraphrase_7b.png'))
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_lcs_reverse', alpha=0.6)
plt.title('Normalized LCS Length vs. Popularity Score (Reverse Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized LCS Length')
plt.savefig(os.path.join(DATASET_DIR, 'normalized_lcs_reverse_7b.png'))
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_levenshtein_paraphrase', alpha=0.6)
plt.title('Normalized Levenshtein Distance vs. Popularity Score (Paraphrase Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized Levenshtein Distance')
plt.savefig(os.path.join(DATASET_DIR, 'normalized_levenshtein_paraphrase_7b.png'))
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='popularity', y='normalized_levenshtein_reverse', alpha=0.6)
plt.title('Normalized Levenshtein Distance vs. Popularity Score (Reverse Prompting)')
plt.xlabel('Popularity Score')
plt.ylabel('Normalized Levenshtein Distance')
plt.savefig(os.path.join(DATASET_DIR, 'normalized_levenshtein_reverse_7b.png'))
plt.close()

# Print correlation information
corr_lcs_p = data['popularity'].corr(data['normalized_lcs_paraphrase'])
corr_lev_p = data['popularity'].corr(data['normalized_levenshtein_paraphrase'])
print(f"Paraphrase Prompting - Correlation between Popularity and Normalized LCS Length: {corr_lcs_p:.4f}")
print(f"Paraphrase Prompting - Correlation between Popularity and Normalized Levenshtein Distance: {corr_lev_p:.4f}")

corr_lcs_r = data['popularity'].corr(data['normalized_lcs_reverse'])
corr_lev_r = data['popularity'].corr(data['normalized_levenshtein_reverse'])
print(f"Reverse Prompting - Correlation between Popularity and Normalized LCS Length: {corr_lcs_r:.4f}")
print(f"Reverse Prompting - Correlation between Popularity and Normalized Levenshtein Distance: {corr_lev_r:.4f}")

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/sy_dataset_1.csv")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of numeric features: {df.shape[1]}")

# Validate 'fraud' column existence
if 'fraud' not in df.columns:
    raise ValueError("DataFrame must contain a 'fraud' column for coloring")

# Extract numeric features excluding 'fraud'
features = df.drop(columns=["claim_id", "claim_date", "fraud"])
features = pd.get_dummies(features, dummy_na=True).fillna(0)

print(f"Number of samples: {features.shape[0]}")
print(f"Number of numeric features: {features.shape[1]}")
print(features.head())


# Run t-SNE projection
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
embedding = tsne.fit_transform(features)


# Plotting with color by 'fraud' label
plt.figure(figsize=(10, 7))
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=df['fraud'], palette='deep', s=50, alpha=0.8)
plt.title("t-SNE projection colored by fraud label")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Fraud")
plt.show()

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Download Microsoft's stock data for the last 100 days
ticker = "MSFT"
data = yf.download(ticker, period="100d")

# Calculate True Range (TR)
data["High-Low"] = data["High"] - data["Low"]
data["High-Close"] = (data["High"] - data["Close"].shift()).abs()
data["Low-Close"] = (data["Low"] - data["Close"].shift()).abs()
data["True Range"] = data[["High-Low", "High-Close", "Low-Close"]].max(axis=1)

# Compute ATR (14-day rolling average of TR)
data["ATR"] = data["True Range"].rolling(window=14).mean()

# Drop NaN values resulting from rolling calculations
data.dropna(inplace=True)

# Reshape ATR values for K-Means clustering
atr_values = data["ATR"].values.reshape(-1, 1)

# Apply K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(atr_values)

# Assign meaningful labels based on cluster centroids
cluster_order = np.argsort(kmeans.cluster_centers_.flatten())  # Sort clusters by ATR value
labels = {cluster_order[0]: "Low", cluster_order[1]: "Medium", cluster_order[2]: "High"}
data["ATR_Category"] = data["Cluster"].map(labels)

# Display categorized ATR data
print(data[["ATR", "ATR_Category"]])

# Save to CSV if needed
data.to_csv("microsoft_atr_kmeans.csv")

# Plot ATR Clustering
plt.scatter(data.index, data["ATR"], c=data["Cluster"], cmap="viridis", label="Clusters")
plt.xlabel("Date")
plt.ylabel("ATR")
plt.title("K-Means Clustering on ATR")
plt.colorbar(label="Cluster")
plt.show()

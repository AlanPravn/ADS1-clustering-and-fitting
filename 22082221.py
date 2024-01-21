# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exponential_growth(x, a, b):
    """
    Exponential growth model function.

    Parameters:
    - x (array-like): Independent variable (e.g., 'Year').
    - a (float): Amplitude parameter.
    - b (float): Growth rate parameter.

    Returns:
    - array-like: Modeled values based on the exponential growth model.
    """
    return a * np.exp(b * (x - x.min()))

# Read the world development indicator data
df = pd.read_csv('WDIData_T.csv')

# Transpose the DataFrame for better visibility
df_transposed = df.transpose()

# Show the transposed data output
print(df_transposed.head())

# Abstract:
# This script performs K-Means clustering on economic indicators, visualizes results,
# and fits an exponential growth model to GDP per capita data.

# ## Clustering

# Select relevant columns for clustering
selected_columns = ['GDP per capita (current US$)',
                    'CO2 emissions (kt)',
                    'GDP (current US$)']
df_selected = df[df['IndicatorName'].isin(selected_columns)]

# Pivot the table for clustering
pivot_table = df_selected.pivot_table(index=['Year', 'CountryCode'],
                                      columns='IndicatorName',
                                      values='Value',
                                      aggfunc='mean')

# Reset the index to avoid potential issues
pivot_table = pivot_table.reset_index()

# Handle missing values
pivot_table = pivot_table.interpolate(method='linear',
                                      axis=0).bfill().ffill()

# Normalize the data if needed (using Min-Max scaling)
X_normalized = (pivot_table[selected_columns] - pivot_table[selected_columns].min()) / (pivot_table[selected_columns].max() - pivot_table[selected_columns].min())

# Define the number of clusters
num_clusters = 3

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
pivot_table['Cluster'] = kmeans.fit_predict(X_normalized)

# Plot the original data with cluster colors
plt.figure(figsize=(12, 8))
colors = ['magenta', 'cyan', 'limegreen']  
markers = ['D', 's', 'o']  
for cluster in range(num_clusters):
    cluster_data = pivot_table[pivot_table['Cluster'] == cluster]
    plt.scatter(cluster_data['Year'],
                cluster_data[selected_columns[0]],
                label=f'Cluster {cluster + 1}',
                color=colors[cluster],
                marker=markers[cluster],
                edgecolors='black',  
                s=120)

plt.title('K-Means Clustering Results')
plt.xlabel('Year')
plt.ylabel(selected_columns[0])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Plot cluster centers
plt.figure(figsize=(12, 8))
for cluster in range(num_clusters):
    cluster_center = kmeans.cluster_centers_[cluster, :]
    plt.scatter(cluster_center[0],
                cluster_center[1],
                marker='X', s=200,
                c='gold',
                label=f'Cluster {cluster + 1}')

plt.scatter(pivot_table['Year'],
            pivot_table[selected_columns[0]],
            c=pivot_table['Cluster'],
            cmap='viridis',
            alpha=0.3)
plt.title('K-Means Clustering Results with Cluster Centers')
plt.xlabel('Year')
plt.ylabel(selected_columns[0])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# ## Fitting

# Select relevant columns for fitting
selected_columns = ['CountryName',
                    'Year',
                    'Value']
df_model = df[df['IndicatorName'] == 'GDP per capita (current US$)'][selected_columns]

# Convert 'Year' to numeric
df_model['Year'] = pd.to_numeric(df_model['Year'],
                                 errors='coerce')

# Drop rows with NaN values in 'Year' or 'Value'
df_model = df_model.dropna(subset=['Year', 'Value'])

# Fit the model using curve_fit
params, covariance = curve_fit(exponential_growth,
                               df_model['Year'],
                               df_model['Value'])

# Model predictions
y_fit = exponential_growth(df_model['Year'], *params)

# Estimate confidence intervals using a constant value for illustration
confidence_interval = 50

# Plotting for exponential growth model
plt.figure(figsize=(12, 8))
plt.scatter(df_model['Year'],
            df_model['Value'],
            label='Original Data',
            color='purple',
            marker='o',
            edgecolors='black',
            s=80)  

plt.plot(df_model['Year'],
         y_fit,
         label='Exponential Growth Fit',
         color='limegreen',
         linestyle='--', 
         linewidth=2) 

# Plot confidence interval
plt.fill_between(df_model['Year'],
                 y_fit - confidence_interval,
                 y_fit + confidence_interval,
                 color='limegreen',
                 alpha=0.3,
                 label='Confidence Interval')
plt.title('Exponential Growth Model Fit with Confidence Interval')
plt.xlabel('Year')
plt.ylabel('GDP per capita (current US$)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
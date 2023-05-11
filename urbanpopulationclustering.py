import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# read in CSV file and skip first 4 rows
Urban_Population = pd.read_csv('UrbanPopulation.csv',skiprows=4)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Display initial five rows
Urban_Population.head(5)

Urban_Population.value_counts("Indicator Name")

Urban_Population.columns.tolist()

# Data types describe dataset attributes
Urban_Population.dtypes

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

missing_values_table(Urban_Population)

# Drop multiple columns from dataset
Urban_Population.drop([ 'Unnamed: 66'], axis=1, inplace= True)

Urban_Population.head(3)

# Precisely impute missing data professionally.
Urban_Population = Urban_Population.fillna(Urban_Population.mean(numeric_only=True))
Urban_Population.isnull().sum().sum()

Urban_Population.tail(10)

# Identify relevant analytical variables concisely.
variables = ['Urban population']
# Filter dataset by specific criteria.
Urban_Population = Urban_Population[(Urban_Population['Indicator Name'].isin(variables))]
Urban_Population.head()

Urban_Population.columns

from sklearn.preprocessing import RobustScaler
# Succinct Column Headings
columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code',
        '1973', '1983', '1993', '2003', '2013', '2020']

# Select columns for new dataframe
df = Urban_Population[columns].copy()

# DataFrame: Index by Country Name
df.set_index('Country Name', inplace=True)

# Columns needing data normalization identified
norm_columns = ['1973', '1983', '1993', '2003', '2013', '2020']

#Normalize dataset for uniform analysis.
scaler = RobustScaler()
df[norm_columns] = scaler.fit_transform(df[norm_columns])

columns = ['Country Code', 'Indicator Name', 'Indicator Code',
        '1973', '1983', '1993', '2003', '2013', '2020']

# apply KMeans clustering
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(df[norm_columns])

# Initializing Clusters with Assigned Values
for i in range(kmeans.n_clusters):
    print(f'Cluster {i}:')
    print(df[df['Cluster'] == i][columns])

from sklearn.metrics import silhouette_score

score = silhouette_score(df[norm_columns], df['Cluster'])
print(f"Silhouette score: {score}")

# Cluster data, plot centroids proficiently.
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df[norm_columns[2]], df[norm_columns[-3]], c=df['Cluster'])
centers = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, -1], marker='o', s=200, linewidths=3, color='r')
ax.set_xlabel('1983')
ax.set_ylabel('2003')
ax.set_title('Clustering using KMeans Algorithm')
plt.show()

# Categorize nations for data analysis.

# Proficiently import climate data.
climate_data_Urban_Population = pd.read_csv('UrbanPopulation.csv', skiprows=4)
climate_data_Urban_Population = climate_data_Urban_Population.fillna(climate_data_Urban_Population.mean(numeric_only=True))

# Choose columns for data analysis
cols = ['1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980',
       '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989',
       '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998',
       '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',
       '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016',
       '2017', '2018', '2019']
data = climate_data_Urban_Population[cols]
data.head(5)

# Normalize data using robust methods.
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)

scaled_data

# Execute KMeans algorithm with 7 clusters
kmeans = KMeans(n_clusters=7, random_state=42).fit(scaled_data)

# Cluster label data assignments professionally.
climate_data_Urban_Population['Cluster'] = kmeans.labels_

# Show country count by clusters
print(climate_data_Urban_Population.groupby('Cluster')['Country Name'].count())

# Choose one nation per cluster
sample_countries = climate_data_Urban_Population.groupby('Cluster').apply(lambda x: x.sample(1))

# Comparative cluster analysis across nations
Cluster_0_urban_population = climate_data_Urban_Population[climate_data_Urban_Population['Cluster'] == 0]
print(Cluster_0_urban_population[cols].mean())


Cluster_0_urban_population

# Cross-national Cluster Comparison Analysis
Cluster_1_urban_population = climate_data_Urban_Population[climate_data_Urban_Population['Cluster'] == 1]
print(Cluster_1_urban_population[cols].mean())

import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Create an LDA scatter plot
lda = LDA(n_components=2)
lda_data = lda.fit_transform(scaled_data, kmeans.labels_)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'orange', '"tab:red', "tab:gray"]
for i in range(5):
    plt.scatter(lda_data[kmeans.labels_==i,0], lda_data[kmeans.labels_==i,1], color=colors[i])
plt.title('Linear Discriminant Analysis')
plt.xlabel('LD_1')
plt.ylabel('LD_2')
plt.show()
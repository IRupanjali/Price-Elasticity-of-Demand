###1. Data Loading & Preview 

import pandas as pd
# Load CSV into Spark DataFrame
df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/rupanjalisingh10@gmail.com/Competition_Data-3.csv")

# Display full Spark DataFrame
df1.display()

# Convert to Pandas
df = df1.toPandas()

# Convert Fiscal Week to datetime
df['Fiscal_Week_ID'] = pd.to_datetime(df['Fiscal_Week_ID'].astype(str) + '-1', format='%Y-%W-%w', errors='coerce')

# Show all data
display(df)

###2. Data Cleaning & Setup
import numpy as np

# Replace inf/-inf with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Convert numeric columns to float
numeric_cols = ['Price', 'Item_Quantity', 'Sales_Amount', 'Sales_Amount_No_Discount', 'Competition_Price']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Display cleaned data
display(df)


###3. Price Elasticity of Demand (PED)
# Sort for correct pct_change calculation
df = df.sort_values(by=['Store_ID', 'Item_ID', 'Fiscal_Week_ID'])

# Compute % changes
df['Price_Change'] = df.groupby(['Store_ID', 'Item_ID'])['Price'].pct_change()
df['Quantity_Change'] = df.groupby(['Store_ID', 'Item_ID'])['Item_Quantity'].pct_change()
df['PED'] = df['Quantity_Change'] / df['Price_Change']

# Clean PED
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['PED'], inplace=True)

# Display PED-added data
display(df)


### 4. EDA Visualizations
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Distributions
px.histogram(df, x='Price', title='Price Distribution').show()
px.histogram(df, x='Item_Quantity', title='Quantity Distribution').show()
px.histogram(df, x='PED', title='PED Distribution').show()

# Scatter
px.scatter(df, x='Price_Change', y='Quantity_Change', title='Price Change vs Quantity Change').show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols + ['PED']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

###5. PED Segmentation
def classify_ped(ped):
    if ped > 1:
        return 'Highly Elastic'
    elif 0 < ped < 1:
        return 'Inelastic'
    elif ped == 1:
        return 'Unitary Elastic'
    elif ped == 0:
        return 'Zero Elasticity'
    else:
        return 'Negative Elasticity'

df['PED_Segment'] = df['PED'].apply(classify_ped)

# Display PED segmentation
display(df[['Store_ID', 'Item_ID', 'PED', 'PED_Segment']])

# Segment counts
segment_counts = df['PED_Segment'].value_counts().reset_index()
px.bar(segment_counts, x='index', y='PED_Segment', title='PED Segmentation',
       labels={'index': 'Segment', 'PED_Segment': 'Count'}).show()

###6. Store-Level Analysis
store_agg = df.groupby('Store_ID').agg({
    'PED': 'mean',
    'Item_Quantity': 'sum'
}).reset_index()

# Display store-level stats
display(store_agg)

# Visualization
px.scatter(store_agg, x='Store_ID', y='PED', size='Item_Quantity',
           title='Store-level Avg PED vs Quantity Sold').show()

###7. Time-Based Trends
time_df = df.groupby('Fiscal_Week_ID').agg({
    'PED': 'mean',
    'Item_Quantity': 'sum',
    'Price': 'mean'
}).reset_index()

# Display time trends
display(time_df)

# Trend plots
px.line(time_df, x='Fiscal_Week_ID', y='PED', title='Average PED Over Time').show()
px.line(time_df, x='Fiscal_Week_ID', y='Item_Quantity', title='Quantity Sold Over Time').show()
px.line(time_df, x='Fiscal_Week_ID', y='Price', title='Average Price Over Time').show()

###8. Outlier Detection
q_low = df['PED'].quantile(0.01)
q_hi = df['PED'].quantile(0.99)
outliers = df[(df['PED'] < q_low) | (df['PED'] > q_hi)]

# Display outliers
display(outliers[['Store_ID', 'Item_ID', 'PED']])

###9. Interactive Filtering in Databricks
# Widget for store filter
dbutils.widgets.dropdown("store_id", "ALL", ["ALL"] + list(map(str, df['Store_ID'].unique())))
selected_store = dbutils.widgets.get("store_id")

# Apply filter
if selected_store != "ALL":
    filtered_df = df[df['Store_ID'] == int(selected_store)]
else:
    filtered_df = df

# Display filtered dataset
display(filtered_df)

# Filtered PED plot
px.histogram(filtered_df, x='PED', title=f"PED Distribution for Store {selected_store}").show()



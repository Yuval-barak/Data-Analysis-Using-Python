import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# נטען את הנתונים מהקבצים שנוצרו
df1 = pd.read_csv('df1.csv')
df2 = pd.read_csv('df2.csv')

# #1
df2_clean = df2.dropna(subset=['Horsepower', 'Rating'])
plt.figure(figsize=(10, 6))
plt.scatter(df2_clean['Horsepower'], df2_clean['Rating'], c='blue')
plt.xlabel('Horsepower')
plt.ylabel('Rating')
plt.title('Scatter Plot of Horsepower vs Rating')
plt.savefig('figure_1.png')
plt.show()

# #2
df2_clean = df2.dropna()
sns.set_style("whitegrid")
sns.pairplot(df2_clean, hue='Fuel', height=3)
plt.suptitle('Pairplot of df2', y=1.02)
plt.savefig('figure_2.png')
plt.show()

# #3
df1_clean = df1.dropna(subset=['Year', 'Price'])
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Year', y='Price', hue='Brand', data=df1_clean)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Scatter Plot of Year vs Price with Brand')
plt.savefig('figure_3.png')
plt.show()

# #4
plt.figure(figsize=(10, 6))
sns.violinplot(x='Fuel', y='Horsepower', data=df2.dropna(subset=['Fuel', 'Horsepower']))
plt.title('Violin Plot of Horsepower by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Horsepower')
plt.savefig('figure_4.png')
plt.show()

# #5
plt.figure(figsize=(10, 6))
sns.boxplot(x='Engine_Type', y='Price', data=df1.dropna(subset=['Engine_Type', 'Price']))
plt.title('Boxplot of Price by Engine Type')
plt.xlabel('Engine Type')
plt.ylabel('Price')
plt.xticks(rotation=90)
plt.savefig('figure_5.png')
plt.show()

# #6
plt.figure(figsize=(10, 6))
sns.kdeplot(df2[df2['Fuel'] == 'Petrol']['Horsepower'].dropna(), label='Petrol', fill=True)
sns.kdeplot(df2[df2['Fuel'] == 'Diesel']['Horsepower'].dropna(), label='Diesel', fill=True)
plt.title('KDE Plot of Horsepower by Fuel Type')
plt.xlabel('Horsepower')
plt.legend()
plt.savefig('figure_6.png')
plt.show()

# #7
plt.figure(figsize=(10, 6))
sns.countplot(x='Transmission', data=df2.dropna(subset=['Transmission']))
plt.title('Countplot of Transmission Types')
plt.xlabel('Transmission Type')
plt.ylabel('Count')
plt.savefig('figure_7.png')
plt.show()

# #8
plt.figure(figsize=(10, 6))
df2_clean = df2.dropna(subset=['Fuel', 'Rating'])
df2_clean.groupby('Fuel')['Rating'].mean().plot(kind='bar', color='purple')
plt.title('Bar Plot of Average Rating by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Average Rating')
plt.savefig('figure_8.png')
plt.show()

# #9
plt.figure(figsize=(10, 6))
df2['Horsepower'].dropna().plot(kind='hist', bins=30, color='teal', edgecolor='black')
plt.title('Histogram of Horsepower')
plt.xlabel('Horsepower')
plt.ylabel('Frequency')
plt.savefig('figure_9.png')
plt.show()

# #10
plt.figure(figsize=(10, 6))
df1_clean = df1.dropna(subset=['Year', 'Price'])
df1_clean.groupby('Year')['Price'].mean().plot(kind='line', marker='o', color='blue')
plt.title('Line Plot of Average Price by Year')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.savefig('figure_10.png')
plt.show()

# #11
merged_df = pd.merge(df1, df2, on='Brand', how='inner')
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rating', y='Price', hue='Fuel', data=merged_df)
plt.xlabel('Rating')
plt.ylabel('Price')
plt.title('Scatter Plot of Rating vs Price with Fuel Type')
plt.savefig('figure_11.png')
plt.show()

# #12
plt.figure(figsize=(10, 6))
sns.boxplot(x='Transmission', y='Horsepower', data=df2.dropna(subset=['Transmission', 'Horsepower']))
plt.title('Box Plot of Horsepower by Transmission Type')
plt.xlabel('Transmission Type')
plt.ylabel('Horsepower')
plt.savefig('figure_12.png')
plt.show()

# #13
plt.figure(figsize=(10, 6))
sns.countplot(y='Brand', data=df1, order=df1['Brand'].value_counts().index)
plt.title('Countplot of Brands')
plt.xlabel('Count')
plt.ylabel('Brand')
plt.savefig('figure_13.png')
plt.show()

# #14
df1_numeric = df1.select_dtypes(include=['float64', 'int64'])
scatter_matrix(df1_numeric, figsize=(10, 10), diagonal='kde', alpha=0.2)
plt.suptitle('Scatter Matrix of df1')
plt.tight_layout()
plt.savefig('figure_14.png')
plt.show()

# #15
# נבצע מיזוג של df1 ו- df2
merged_df = pd.merge(df1, df2, on='Brand', how='inner')

# נוודא שאין ערכים חסרים בעמודות הרלוונטיות
merged_df = merged_df.dropna(subset=['Horsepower', 'Rating', 'Price', 'Fuel'])

# יצירת גרף תלת ממדי
fig = px.scatter_3d(merged_df, x='Horsepower', y='Rating', z='Price', color='Fuel',
                    title='3D Scatter Plot of Horsepower, Rating, and Price by Fuel Type',
                    labels={'Horsepower': 'Horsepower', 'Rating': 'Rating', 'Price': 'Price'},
                    hover_name='Brand')

# שמירת הגרף כקובץ
fig.write_html("figure_15.html")
fig.show()

# שלב 4.1: חלוקה לאשכולות
# ניקוי נוסף של הנתונים לאחר המיזוג
merged_df = merged_df.dropna(subset=['Year', 'Price', 'Horsepower', 'Rating'])

data = merged_df[['Year', 'Price']]

# נרמול הנתונים
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# פונקציה לחישוב WCSS
def calculate_wcss(data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

wcss = calculate_wcss(data_scaled)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('figure_elbow_method.png')
plt.show()

optimal_clusters = 3  # על פי מבחן המרפק מהגרף
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
merged_df['Cluster'] = kmeans.fit_predict(data_scaled)

plt.figure(figsize=(10, 5))
sns.scatterplot(x='Year', y='Price', hue='Cluster', data=merged_df, palette='viridis')
plt.title('Clusters Visualization')
plt.savefig('figure_clusters.png')
plt.show()

# שלב 4.2: רגרסיה ליניארית
# בחירת עמודות רלוונטיות לרגרסיה
X = merged_df[['Horsepower']]
y = merged_df['Price']

# התאמת המודל
regressor = LinearRegression()
regressor.fit(X, y)

# חיזוי ערכים
y_pred = regressor.predict(X)

# ציור גרף הרגרסיה
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.title('Linear Regression: Horsepower vs. Price')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.savefig('figure_regression.png')
plt.show()

# הצגת מקדמי הרגרסיה
intercept = regressor.intercept_
coefficient = regressor.coef_

print(f'Intercept: {intercept}')
print(f'Coefficient: {coefficient}')

# שלב 4.3: ניתוח נוסף באמצעות Random Forest Regressor
features = merged_df[['Year', 'Horsepower', 'Rating']]
target = merged_df['Price']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title('Random Forest Regression: Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.savefig('figure_rf_regression.png')
plt.show()

# ניתוח קורלציה
correlation_matrix = merged_df[['Year', 'Horsepower', 'Rating', 'Price']].corr()

plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.savefig('figure_correlation_matrix.png')
plt.show()

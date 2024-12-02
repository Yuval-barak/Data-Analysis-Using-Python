import pandas as pd

input1_df = pd.read_csv('input1_df.csv')
print('The data of the CSV file:')
print(input1_df)

# 1.3 הדפסת מידע על מערך הנתונים
print("\nColumns names:")
print(input1_df.columns)

print('\nData types and missing values:')
print(input1_df.info())

#1.4
#get summary of continuos variables
print('\nThe summary of continuos variables')
print(input1_df.describe())

#get summary of continuos and categorical variables
print('\n The summary of continuos and categorical variables')
print(input1_df.describe(include='all'))

num_columns = len(input1_df.columns)
print("\nNumber of columns in the DataFrame:", num_columns)

# כמה רשומות יש במערך הנתונים
num_rows = len(input1_df)
print("Number of rows in the DataFrame:", num_rows)

# מהם סוגי הנתונים בכל עמודה
data_types = input1_df.dtypes
print("Data types of each column:\n", data_types)

# חברות מכוניות פופולריות
popular_brands = input1_df['Brand'].value_counts().head(10)
print("\nTop 10 popular car brands:")
print(popular_brands)

# מחיר מכירה ממוצע לפי חברה
average_selling_price_by_brand = input1_df.groupby('Brand')['Price'].mean()
print("\nAverage selling price by brand:")
print(average_selling_price_by_brand)

# סוג הדלק
fuel_distribution = input1_df['Fuel'].value_counts()
print("\nFuel type distribution:")
print(fuel_distribution)

#1.5
# ספירה וחישוב אחוזים של מספר הערכים החסרים בכל עמודה
missing_values_count = input1_df.isnull().sum()
percentage_missing_values = (missing_values_count / len(input1_df)) * 100

print("\nPercentage of missing values for each column:")
print(percentage_missing_values)

# חישוב סך מספר הנתונים החסרים
total_missing_values = missing_values_count.sum()

# חישוב אחוז הנתונים החסרים בקובץ
total_entries = input1_df.size
percentage_total_missing_values = (total_missing_values / total_entries) * 100

print("Total number of missing values in the DataFrame:", total_missing_values)
print("Percentage of missing values in the entire DataFrame:", percentage_total_missing_values)

#1.8
# הדפסת 5 השורות הראשונות
print("The first 5 rows of the DataFrame:")
print(input1_df.head())

# הדפסת 5 השורות האחרונות
print("\nThe last 5 rows of the DataFrame:")
print(input1_df.tail())

# הדפסת 5 כלשהם שורות מאמצע הקובץ
print("\n5 rows from the middle of the DataFrame:")
print(input1_df[480:485])

#1.9
#סטטיסטיקה תיאורית
print("\nDescriptive statistics of the DataFrame:")
print(input1_df.describe(include='all'))

#------------

# קריאת הקובץ input2_df.csv
input2_df = pd.read_csv('input2_df.csv')

print('The data of the CSV file:')
print(input2_df)

# 1.3 הדפסת מידע על מערך הנתונים
print("\nColumns names:")
print(input2_df.columns)

print('\nData types and missing values:')
print(input2_df.info())

# 1.4 קבלת סיכום של משתנים רציפים
print('\nThe summary of continuous variables:')
print(input2_df.describe())

# קבלת סיכום של משתנים רציפים וקטגוריים
print('\nThe summary of continuous and categorical variables:')
print(input2_df.describe(include='all'))

# מספר העמודות במערך הנתונים
num_columns = len(input2_df.columns)
print("\nNumber of columns in the DataFrame:", num_columns)

# מספר הרשומות במערך הנתונים
num_rows = len(input2_df)
print("Number of rows in the DataFrame:", num_rows)

# מהם סוגי הנתונים בכל עמודה
data_types = input2_df.dtypes
print("\nData types of each column:\n", data_types)

# מותגים פופולריים בקובץ
popular_brands = input2_df['Brand'].value_counts().head(10)
print("\nTop 10 popular car brands:")
print(popular_brands)

# הצגת ממוצע דירוג לפי מותג (במידה ויש)
if 'rating' in input2_df.columns:
    average_rating_by_brand = input2_df.groupby('Brand')['rating'].mean()
    print("\nAverage rating by brand:")
    print(average_rating_by_brand)
else:
    print("\n'Rating' column not found in the DataFrame.")

# הצגת תפלגות סוגי המנוע (אם קיימת)
if 'engine_type' in input2_df.columns:
    engine_type_distribution = input2_df['engine_type'].value_counts()
    print("\nEngine type distribution:")
    print(engine_type_distribution)
else:
    print("\n'Engine_Type' column not found in the DataFrame.")

# 1.5 ספירה ואחוזים של מספר הערכים החסרים בכל עמודה
missing_values_count = input2_df.isnull().sum()
percentage_missing_values = (missing_values_count / len(input2_df)) * 100

print("\nPercentage of missing values for each column:")
print(percentage_missing_values)

# חישוב סך מספר הנתונים החסרים
total_missing_values = missing_values_count.sum()

# חישוב אחוז הנתונים החסרים בקובץ
total_entries = input2_df.size
percentage_total_missing_values = (total_missing_values / total_entries) * 100

print("Total number of missing values in the DataFrame:", total_missing_values)
print("Percentage of missing values in the entire DataFrame:", percentage_total_missing_values)


#1.11
# ייבוא ה-DataFrames
input1_df = pd.read_csv('input1_df.csv')  # נתוני רכב
input2_df = pd.read_csv('input2_df.csv')  # ביקורות רכב

#מיזוג
outer_join_df = input1_df.merge(input2_df, on='Brand', how='outer')

# הדפסת 5 השורות הראשונות מה-DataFrame הממוזג
print(outer_join_df.head())

# בדיקת ערכים חסרים ב-DataFrame הממוזג
missing_values_count = outer_join_df.isnull().sum()
print("\nMissing values in the merged DataFrame:")
print(missing_values_count)

# חישוב אחוזי הערכים החסרים
percentage_missing_values = (missing_values_count / len(outer_join_df)) * 100
print("\nPercentage of missing values in the merged DataFrame:")
print(percentage_missing_values)

# סך כל הערכים החסרים במערך הנתונים הממוזג
total_missing_values = missing_values_count.sum()
print("Total number of missing values in the merged DataFrame:", total_missing_values)

# אחוז הערכים החסרים במערך הנתונים הממוזג כולו
percentage_total_missing = (total_missing_values / (len(outer_join_df) * len(outer_join_df.columns))) * 100
print("Percentage of missing values in the entire merged DataFrame:", percentage_total_missing)

# # שמירת הקובץ הממוזג
# merged_df.to_csv('merged_car_data.csv', index=False)
#
# # הצגת הנתונים הממוזגים
# print('The merged data of the CSV file:')
# print(merged_df)

#1.12
# בחירת עמודות עבור df1
df1 = outer_join_df[['Brand', 'Model', 'Year', 'Engine_Type', 'Price']]

# בחירת עמודות עבור df2
df2 = outer_join_df[['Brand', 'Rating', 'Fuel', 'Horsepower', 'Transmission']]

df1.to_csv('df1.csv', index=False)
df2.to_csv('df2.csv', index=False)

# הדפסת ה-DataFrame החדשים
print("\nDataFrame df1:")
print(df1.head())

print("\nDataFrame df2:")
print(df2.head())

#---------
import pandas as pd
import numpy as np

# קריאת הקבצים המקוריים
df1 = pd.read_csv('df1.csv')
df2 = pd.read_csv('df2.csv')

#----------------------------------2.1
#תיקון נתונים לא מתאימים בעמודות מספריות

# עמודת Year ו-Price ב-df1
for col in ['Year', 'Price']:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')

# עמודת Horsepower ב-df2 אם קיימת
if 'Horsepower' in df2.columns:
    df2['Horsepower'] = pd.to_numeric(df2['Horsepower'], errors='coerce')

print("\nCleaned df1:")
print(df1.head())

print("\nCleaned df2:")
print(df2.head())

#---------------2.2
#תיקון נתונים לא מתאימים בעמודות טקסט
# עמודות טקסט ב-df1
for col in ['Brand', 'Model', 'Engine_Type']:
    df1[col] = df1[col].apply(lambda x: x if isinstance(x, str) else np.nan)

# עמודות טקסט ב-df2
for col in ['Brand', 'Fuel', 'Transmission']:
    df2[col] = df2[col].apply(lambda x: x if isinstance(x, str) else np.nan)

print("\nFixed text columns in df1:")
print(df1.head())

print("\nFixed text columns in df2:")
print(df2.head())

#---------------2.3
#טיפול בערכים חסרים
# מילוי ערכים חסרים בממוצע, השכיח או החציון בעמודות מספריות וטקסט
df1['Year'] = df1['Year'].fillna(df1['Year'].median())
df1['Price'] = df1['Price'].fillna(df1['Price'].mean())
df1['Model'] = df1['Model'].fillna(df1['Model'].mode()[0])
df1['Engine_Type'] = df1['Engine_Type'].fillna(df1['Engine_Type'].mode()[0])

df2['Rating'] = df2['Rating'].fillna(df2['Rating'].mean())
df2['Horsepower'] = df2['Horsepower'].fillna(df2['Horsepower'].median())
df2['Fuel'] = df2['Fuel'].fillna(df2['Fuel'].mode()[0])
df2['Transmission'] = df2['Transmission'].fillna(df2['Transmission'].mode()[0])

print("\nMissing values after handling in df1:")
print(df1.isnull().sum())

print("\nMissing values after handling in df2:")
print(df2.isnull().sum())

#---------------2.4
#נרמול עמודות מספריות
# Normalizing 'Year' and 'Price' columns in df1
df1['Year_norm'] = df1['Year'] / df1['Year'].max()
df1['Price_norm'] = df1['Price'] / df1['Price'].max()

# Normalizing 'Horsepower' column in df2 if exists
if 'Horsepower' in df2.columns:
    df2['Horsepower_norm'] = df2['Horsepower'] / df2['Horsepower'].max()

print("\nNormalized df1:")
print(df1.head())

print("\nNormalized df2:")
print(df2.head())

#----------------2.5
#הדפסת שורות כפולות
# הדפסת שורות כפולות בdf1 ומחיקת כפילויות
df1_duplicates = df1[df1.duplicated()]
print("\nDuplicate rows in df1:")
print(df1_duplicates)

df1.drop_duplicates(inplace=True)
df1.reset_index(drop=True, inplace=True)

print("\nDataFrame df1 without duplicates:")
print(df1.head())

# הדפסת שורות כפולות בdf2 ומחיקת כפילויות
df2_duplicates = df2[df2.duplicated()]
print("\nDuplicate rows in df2:")
print(df2_duplicates)

df2.drop_duplicates(inplace=True)
df2.reset_index(drop=True, inplace=True)

print("\nDataFrame df2 without duplicates:")
print(df2.head())

# שמירת הנתונים המעודכנים חזרה ל-CSV
df1.to_csv('df1.csv', index=False)
df2.to_csv('df2.csv', index=False)

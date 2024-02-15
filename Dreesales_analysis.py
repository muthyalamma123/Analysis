#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the required Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Data Cleaning 

# In[2]:


### Data Reading & Data Types 


# In[3]:


import pandas as pd
#Read the data in pandas
inp0= pd.read_csv(r'C:\Users\Windows\Downloads\Data_Analysis_Project\Data_Analysis_Project\Attribute DataSet.csv')
inp1= pd.read_csv(r'C:\Users\Windows\Downloads\Data_Analysis_Project\Data_Analysis_Project\Dress Sales.csv')
inp0


# In[5]:


inp1


# You have “Attribute DataSet” which contains a column named “Price”. Choose the correct statement from the following about its data type and variable type.
# - Integer type and numerical variable
# - Object type and categorical ordinal variable
# - Object type and categorical nominal variable
# #- Float type and categorical variable.
# 

# 
# 
# There is another column in “Attribute DataSet” named as “Recommendation”, choose the correct statement about its data type and variable type.
# - Integer type and categorical
# #- Object type and categorical
# - Integer type and continuous numerical
# - Object type only.
# 

# 
# Which of the following column do you think are of no use in “Attribute DataSet”.
# - Dress_ID
# - Price
# - Size and material
# - NeckLine
# #- None of the above
# 

# In[6]:


# Print the information about the attributes of inp0 and inp1.
inp0.info


# In[7]:




inp1.info


# ### Fixing the Rows and Columns 

# As you can see, there is a column in “Attribute Dataset” named as ‘Size’. This column contains the values in abbreviation format. Write a code in Python to convert the followings:
# 
# - M into  “Medium”
# - L into  “Large”
# - XL into “Extra large”
# - free into “Free”
# - S, s & small into “Small”.
# 
# Now once you are done with changes in the dataset, what is the value of the lowest percentage, the highest percentage and the percentage of Small size categories in the column named “Size”?
# 

# In[8]:


inp0.Size.value_counts()


# In[9]:


## Column fixing, correcting size abbreviation. Count the percentage of each size category in "Size" column.
changename = lambda n: 'Medium' if n == 'M' else ('Large' if n == 'L' else ('Extra Large' if n == 'XL' else ('Free' if n == 'free' else 'Small')))
inp0.head()


# In[10]:


# Print the value counts of each category in "Size" column.
inp0.Size.value_counts()/len(inp0)*100
inp0.Size.value_counts(normalize=True)


# ### Impute/Remove Missing values

# In[11]:


# Print the null count of each variables of inp0 and inp1.


# In[12]:


# Print null count for inp0
print("Null count for inp0:")
print(inp0.isnull().sum())


# In[13]:


# Print null count for inp1
print("\nNull count for inp1:")
print(inp1.isnull().sum())


# In[14]:


# Print null count for inp1
print("Null count for inp1:")
print(inp1.isnull().sum())


# You are given another dataset named “Dress Sales”. Now if you observe the datatypes of the columns using ‘inp1.info()’ command, you can identify that there are certain columns defined as object data type though they primarily consist of numeric data.
# 
# Now if you try and convert these object data type columns into numeric data type(float), you will come across an error message. Try to correct this error.
# 
# 
# 
# 
# 
# 

# In[24]:


# Print the data types information of inp1 i.e. "Dress Sales" data.
inp1.info()


# In[16]:


# Try to convert the object type into float type of data. YOU GET ERROR MESSAGE.
inp1['16-09-2013']=inp1['16-09-2013'].astype(float)


# In[17]:


inp1.head()


# In[23]:


# Do the required changes in the "Dress Sales" data set to get null values on string values.
print(inp1.info())


# In[19]:


# Convert the object type columns in "Dress Sales" into float type of data type.
(inp1.isnull().sum()/len(inp1))*100


# In[20]:


# Convert the object type columns in "Dress Sales" into float type of data type.
inp1['16-09-2013'] =pd.to_numeric(inp1['16-09-2013'], errors='coerce')
inp1.dtypes


# In[21]:


# Convert the object type columns in "Dress Sales" into float type of data type.
(inp1.isnull().sum()/len(inp1))*100


# When you see the null counts in “Dress Sales” dataset after performing all the operations that have been mentioned in jupyter notebook, you will find that there are some columns in “Dress Sales” data where there are more than 40% of missing values. Based on your understanding of dealing with missing values do the following steps.

# In[ ]:


#print the null percentage of each column of inp1


# In[22]:



# Assuming 'inp1' is your DataFrame
# Replace 'inp1' with the actual name of your DataFrame

# Calculate null percentage for each column
null_percentage = inp1.isnull().mean() * 100

# Print the null percentage of each column
print("Null Percentage of each column in inp1:")
print(null_percentage)


# In[36]:


# Drop the columns in "Dress Sales" which have more than 40% of missing values.
a=inp1.isnull().mean()*100
column=a[a>40].index
inp1=inp1.drop(columns=column)
inp1.isnull().sum()


# You should categorise the dates into seasons in “Dress Sales” data to simplify the analysis according to the following criteria:
# - June, July and August: Summer.
# - September, October and November: Autumn.
# - December, January and February: WInter.
# - March, April and May: Spring.
# 
# 
# 

# In[38]:


inp1.head()


# In[39]:


# Create the four seasons columns in inp1, according to the above criteria.
summer=(inp1[['29-08-2013','31-08-2013','09-06-2013','09-08-2013', '10-06-2013']].sum()).sum()
Winter = (inp1[['09-02-2013','10-12-2013']].sum()).sum()
Spring = (inp1['09-04-2013'].sum()).sum()
Autumn= ['09-10-2013', '14-09-2013', '16-09-2013', '18-09-2013', '20-09-2013', '22-09-2013', '24-09-2013', '28-09-2013']
inp1[Autumn] = inp1[Autumn].apply(pd.to_numeric, errors='coerce')
Autumn = inp1[Autumn].sum().sum()
print('summer:-',summer)
print('Autumn:-',Autumn)
print('Winter:-',Winter)
print('Spring:-',Spring)


# In[40]:


# Create the four seasons columns in inp1, according to the above criteria.
inp1["Summer"]=inp1["29-08-2013"]+inp1["31-08-2013"]+inp1["09-06-2013"]+inp1["09-08-2013"]+inp1["10-06-2013"]
inp1["Autumn"]=inp1["09-10-2013"]+inp1["14-09-2013"]+inp1["16-09-2013"]+inp1["18-09-2013"]+inp1["20-09-2013"]+inp1["22-09-2013"]+inp1["24-09-2013"]+inp1["28-09-2013"]
inp1["Winter"]=inp1["09-02-2013"]+inp1["09-12-2013"]+inp1["10-12-2013"]
inp1["Spring"]=inp1["09-04-2013"]
inp1.head()


# In[41]:


import pandas as pd

# Simplified data creation
data = {'Date': ['29-08-2013', '31-08-2013', '09-02-2013', '09-04-2013',
                 '09-06-2013', '09-08-2013', '09-10-2013', '09-12-2013',
                 '14-09-2013', '16-09-2013', '18-09-2013', '20-09-2013',
                 '22-09-2013', '24-09-2013', '28-09-2013', '10-06-2013',
                 '10-12-2013']}
df = pd.DataFrame({'Date': pd.to_datetime(data['Date'], format='%d-%m-%Y')})

# Create a new 'Season' column based on the month
df['Season'] = df['Date'].dt.month.map({1: 'Winter', 2: 'Winter', 3: 'Spring',
                                       4: 'Spring', 5: 'Spring', 6: 'Summer',
                                       7: 'Summer', 8: 'Summer', 9: 'Autumn',
                                       10: 'Autumn', 11: 'Autumn', 12: 'Winter'})

# Display the DataFrame with the new 'Season' column
print(df.head(50))


# In[42]:


# calculate the sum of sales in each seasons in inp1 i.e. "Dress Sales".
inp1.sum()


# Now let's merge inp1 with inp0 with left join manner, so that the information of inp0 should remain intact.

# In[43]:


inp1['Spring'] = inp1.apply(lambda x: x['09-04-2013'], axis=1)

inp1['Summer'] = inp1.apply(lambda x: x['29-08-2013'] + x['31-08-2013']+ x['09-06-2013']+ x['09-08-2013']+ x['10-06-2013'], axis=1)

inp1['Winter'] = inp1[['09-02-2013', '09-12-2013', '10-12-2013']].sum(axis=1)

inp1['Autumn'] = inp1[['09-10-2013', '14-09-2013', '16-09-2013', '18-09-2013', '20-09-2013', '22-09-2013', '24-09-2013', '28-09-2013']].sum(axis=1, skipna=True)


# In[44]:


import pandas as pd
# Merge inp0 with inp1 into inp0. this is also called left merge.
b = pd.merge(left=inp0,right=inp1, how='left', left_on='Dress_ID', right_on='Dress_ID')
b.head()


# In[45]:


# Example of creating or importing DataFramesolumns
import pandas as pd

# Assuming inp0 and inp1 are your DataFrames
inp0 = pd.DataFrame({'Dress_ID': [1, 2, 3], 'Other_Column': ['A', 'B', 'C']})
inp1 = pd.DataFrame({'Dress_ID': [1, 2, 4], 'Additional_Column': ['X', 'Y', 'Z']})


# In[46]:


print("Columns in inp0:", inp0.columns)
print("Columns in inp1:", inp1.columns)

print("First few rows of inp0:\n", inp0.head())
print("First few rows of inp1:\n", inp1.head())


# In[47]:


import pandas as pd
# Assuming inp0 is your DataFrame with datetime columns
inp0 = pd.DataFrame({
    'Date': pd.date_range(start='2022-01-01', periods=10, freq='D'),
    'Value1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Value2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
})

# Convert 'Date' column to datetime if not already in datetime format
inp0['Date'] = pd.to_datetime(inp0['Date'])

# Drop columns within the specified date range
date_range_to_drop = pd.date_range('2013-08-29', '2013-12-10')
inp0.drop(columns=date_range_to_drop.intersection(inp0.columns), inplace=True)

# Check for null values
null_values = inp0.isnull().sum()

print(null_values)


# Print the null count of inp0 to get the idea about the missing values in data set.

# In[48]:


# Print the null count of each columns in inp0 dataframe i.e. combined data frame of inp0 and inp1 without date columns.
inp0.isnull().sum()


# 
# 
# You can see that there are two types of variables one with a large number of missing values and another is very less number of missing values. These two columns can be categorized as:
# 
# Type-1: Missing values are very less (around 2 or 3 missing values): Price, Season, NeckLine, SleeveLength, Winter and Autumn. 
# 
# Type-2: Missing values are large in numbers (more than 15%): Material, FabricType, Decoration and Pattern Type.
# 
# 

# In[49]:


# Deal with the missing values of Type-1 columns: Price, Season, NeckLine, SleeveLength, Winter and Autumn.
columns_to_fill = ['Season', 'NeckLine', 'SleeveLength']
for column in columns_to_fill:
    if column in inp0.columns:
        inp0[column].fillna(inp0[column].mode()[0], inplace=True)
    if column in inp1.columns:
        inp1[column].fillna(inp1[column].mode()[0], inplace=True)

# Deal with the missing values of Type-2 columns: Winter and Autumn.
columns_to_fill = ['Winter', 'Autumn']
for column in columns_to_fill:
    if column in inp0.columns:
        inp0[column].fillna(inp0[column].mode()[0], inplace=True)
    if column in inp1.columns:
        inp1[column].fillna(inp1[column].mode()[0], inplace=True)

# Display the first few rows of inp0 DataFrame
inp0.head()


# In[51]:


# Deal with the missing values for Type-2 columns: Material, FabricType, Decoration and Pattern Type.
columns_to_fill_type2 = ['Material', 'FabricType', 'Decoration', 'Pattern Type']
for column in columns_to_fill_type2:
    if column in inp0.columns:
        inp0[column].fillna(inp0[column].mode()[0], inplace=True)

# Display the first few rows of inp0 DataFrame
inp0.head


# In[53]:


inp0.columns


# In[54]:


inp0.head()


# ### Standardise value 

# In the given dataset, there are certain discrepancies with the categorical names such as irregular spellings. Choose the correct option of columns with irregular categories and update them.
#  
# - Season, NeckLine
# - Price, Material
# - fabricType, Decoration
# - Season, SleeveLength
# 

# In[55]:


#correcting the spellings.
b.Season= b.Season.replace('Automn', "Autumn")

b.Season= b.Season.replace('spring', "Spring")

b.Season= b.Season.replace('winter', "Winter")
b.head()


# In[56]:


b.columns


# In[57]:


inp0.head()


# There is a column named ‘Style’ in ‘Attribute Dataset’ which consists of the different style categories of the women apparels. Certain categories whose total sale is less than 50000 across all the seasons is considered under one single category as ‘Others’.
# 

# Which of the following categories in ‘Style’ column can be grouped into ‘Others’ category? and perform the grouping operation in the notebook for further analysis.
# - Flare, fashion
# - Novelty, bohemian
# - OL, fashion, work
# - Novelty, fashion, Flare
# 

# In[58]:


# Group "Style" categories into "Others" which have less than 50000 sales across all the seasons.
#inp0['total'] = inp0['Summer'] + inp0['Autumn'] + inp0['Winter'] + inp0['Spring']
b['total'] = b[['Summer', 'Autumn', 'Winter', 'Spring']].sum(axis=1)

# Calculate total sales for each style
style_sales = b.groupby('Style')['total'].sum()

# Identify styles with total sales less than 50000
styles_to_group = style_sales[style_sales < 50000].index.tolist()

# Replace those styles with 'Others' using boolean indexing
b.loc[b['Style'].isin(styles_to_group), 'Style'] = 'Others'

# Check the updated DataFrame
b['Style'].unique()


# In[60]:


# Assuming 'total' is the column representing total sales in your DataFrame
# 'Style' is the column containing style categories
# Assuming 'total' is the column representing total sales in your DataFrame
# 'Style' is the column containing style categories

# Calculate total sales for each style
style_sales = b.groupby('Style')['total'].sum()

# Identify styles with total sales less than 50000
styles_to_group = style_sales[style_sales < 50000].index.tolist()

# Replace those styles with 'Others' using boolean indexing
b.loc[b['Style'].isin(styles_to_group), 'Style'] = 'Others'

# Check the unique values in the 'Style' column
print(b['Style'].unique())


# In[61]:


inp0.head()


# What is the percentage of “cute” and “Others” category in “Style” column in “Attribute DataSet” respectively?
# - 46%, 5%
# - 9%, 2.1%
# - 2.1%, 5%
# - 13.8%, 9%
# 

# In[62]:


# Calculate the percentage of each categories in the "Style" variable.
style_percentage = b['Style'].value_counts(normalize=True) * 100

# Print or use the results as needed
print("Percentage of each category in the 'Style' variable:")
print(style_percentage)


# Similarly Club Neckline, SLeeve length categories into "Others" which have less than 50000 sales across all the seasons.

# In[64]:


# Group "Neckline" categories into "Others" which have less than 50000 sales across all the seasons.
# Calculate total sales for each neckline category
neckline_sales = b.groupby('NeckLine')['total'].sum()

# Identify neckline categories with total sales less than 50000
necklines_to_group = neckline_sales[neckline_sales < 50000].index.tolist()

# Replace those neckline categories with 'Others' using boolean indexing
b.loc[b['NeckLine'].isin(necklines_to_group), 'NeckLine'] = 'Others'

# Check the unique values in the 'NeckLine' column
print(b['NeckLine'].unique())


# In[65]:


# Display the column names
print(inp0.columns)


# In[66]:


# Group "Sleeve length" categories into "Others" which have less than 50000 sales across all the seasons.
# Assuming 'total' is the column representing total sales in your DataFrame
# 'SleeveLength' is the column containing sleeve length categories

# Calculate total sales for each sleeve length category
sleeve_length_sales = b.groupby('SleeveLength')['total'].sum()

# Identify sleeve length categories with total sales less than 50000
sleeve_lengths_to_group = sleeve_length_sales[sleeve_length_sales < 50000].index.tolist()

# Replace those sleeve length categories with 'Others' using boolean indexing
b.loc[b['SleeveLength'].isin(sleeve_lengths_to_group), 'SleeveLength'] = 'Others'

# Check the unique values in the 'SleeveLength' column
print(b['SleeveLength'].unique())


# In[67]:


# Group "material" categories into "Others" which have less than 25000 sales across all the seasons.
# Assuming 'total' is the column representing total sales in your DataFrame
# 'Material' is the column containing material categories

# Calculate total sales for each material category
material_sales = b.groupby('Material')['total'].sum()

# Identify material categories with total sales less than 25000
materials_to_group = material_sales[material_sales < 25000].index.tolist()

# Replace those material categories with 'Others' using boolean indexing
b.loc[b['Material'].isin(materials_to_group), 'Material'] = 'Others'

# Check the unique values in the 'Material' column
print(b['Material'].unique())


# Club material, fabrictype, patterntype and decoration categories into "Others" which have less than 25000 sales across all the seasons

# In[68]:


# Group "fabric type" categories into "Others" which have less than 25000 sales across all the seasons

# Calculate total sales for each fabric type category
fabric_type_sales = b.groupby('FabricType')['total'].sum()

# Identify fabric type categories with total sales less than 25000
fabric_types_to_group = fabric_type_sales[fabric_type_sales < 25000].index.tolist()

# Replace those fabric type categories with 'Others' using boolean indexing
b.loc[b['FabricType'].isin(fabric_types_to_group), 'FabricType'] = 'Others'

# Check the unique values in the 'FabricType' column
print(b['FabricType'].unique())


# In[69]:


# Group "patern type" categories into "Others" which have less than 25000 sales across all the seasons.
# Assuming 'total' is the column representing total sales in your DataFrame
# 'Pattern Type' is the column containing pattern type categories

# Calculate total sales for each pattern type category
pattern_type_sales = b.groupby('Pattern Type')['total'].sum()

# Identify pattern type categories with total sales less than 25000
pattern_types_to_group = pattern_type_sales[pattern_type_sales < 25000].index.tolist()

# Replace those pattern type categories with 'Others' using boolean indexing
b.loc[b['Pattern Type'].isin(pattern_types_to_group), 'Pattern Type'] = 'Others'

# Check the unique values in the 'Pattern Type' column
print(b['Pattern Type'].unique())


# In[70]:


# Group "decoration" categories into "Others" which have less than 25000 sales across all the seasons.
# Assuming 'total' is the column representing total sales in your DataFrame
# 'Decoration' is the column containing decoration categories

# Calculate total sales for each decoration category
decoration_sales = b.groupby('Decoration')['total'].sum()

# Identify decoration categories with total sales less than 25000
decorations_to_group = decoration_sales[decoration_sales < 25000].index.tolist()

# Replace those decoration categories with 'Others' using boolean indexing
b.loc[b['Decoration'].isin(decorations_to_group), 'Decoration'] = 'Others'

# Check the unique values in the 'Decoration' column
print(b['Decoration'].unique())


# ### Caregorical Ordered Univariate Analysis

# Which of the following is an unordered variable in “Attribute DataSet”.
# #- Style
# - Price
# - Season
# - Size
# 

# ### Numerical variable Univariate analysis:

# What is the approximate difference between the maximum value and 75th percentile in “Autumn” column.
# - Approx 54000
# - Approx 55000
# #- Approx 52000
# - Approx 50000
# 
# 

# In[72]:


# Describe the numerical variale: "Autumn".
b.Autumn.describe()


# In[73]:


#plot the boxplot of "Autumn" column.

autumn_column = b['Autumn']
# Create a boxplot
sns.boxplot(b.Autumn) 
plt.title('Boxplot of Autumn Column')
plt.xlabel('Sales')
plt.show()


# In[75]:


inp0.head()


# In[76]:


print(inp0.columns)


# Which of the following season has the highest difference between the maximum value and 99th quantile of sales?
# - Winter
# - Summer
# - Spring
# #- Autumn
# 

# In[77]:


# Find the maximum and 50th percentile of Winter season.
winter_Sales = b['Winter'].to_numpy()

# Calculate the maximum temperature
max_Sales = np.max(winter_Sales)

# Calculate the 50th percentile (median) temperature
percentile_50 = np.percentile(winter_Sales, 50)

# Print the results
print(f"Maximum Sales: {max_Sales}")
print(f"50th percentile Sales: {percentile_50}")


# In[78]:


print(b['Winter'].max())
print(b['Winter'].quantile(0.50))


# In[79]:


# Find the maximum and 50th percentile of Summer season.
winter_Sales = b['Summer'].to_numpy()

# Calculate the maximum temperature
max_Sales = np.max(winter_Sales)

# Calculate the 50th percentile (median) temperature
percentile_50 = np.percentile(winter_Sales, 50)

# Print the results
print(f"Maximum Sales: {max_Sales}")
print(f"50th percentile Sales: {percentile_50}")


# In[80]:


# Find the maximum and 50th percentile of Spring season.


# In[81]:


import numpy as np
# Assuming inp1 is your DataFrame with a "Season" column
inp1 = pd.DataFrame({
    'Date': pd.date_range('2022-01-01', '2022-12-31', freq='M'),
    'Sales': [100, 120, 150, 80, 200, 180, 220, 250, 120, 180, 200, 150],
    'Season': pd.cut(pd.date_range('2022-01-01', '2022-12-31', freq='M').month, bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])
})

# Filter the DataFrame for Spring season
spring_sales = inp1[inp1['Season'] == 'Spring']['Sales']

# Calculate maximum and 50th percentile using numpy
max_sales = np.max(spring_sales)
percentile_50 = np.percentile(spring_sales, 50)

# Print the results
print(f"Maximum sales in Spring: {max_sales}")
print(f"50th percentile of sales in Spring: {percentile_50}")


# In[82]:


# Find the maximum and 50th percentile of Autumn season.
winter_Sales= b['Autumn'].to_numpy()

# Calculate the maximum temperature
max_Sales = np.max(winter_Sales)

# Calculate the 50th percentile (median) temperature
percentile_50 = np.percentile(winter_Sales, 50)

# Print the results
print(f"Maximum Sales: {max_Sales}")
print(f"50th percentile Sales: {percentile_50}")


# In[ ]:


# Find the maximum and 50th percentile of Autumn season.


# In[1]:


Percentile = (Number of Values Below “x” / Total Number of Values) × 100


# In[ ]:





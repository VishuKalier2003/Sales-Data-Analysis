import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
files = [file for file in os.listdir('./Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data')]
data = pd.DataFrame()     # Creating an empty dataframe
for file in files:
    # reading all files location
    df = pd.read_csv('./Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data/'+file)
    # Concatenating them into the basic DataFrame
    data = pd.concat([data, df])
# Indexing false means the first integer index values of the dataframe
data.to_csv('AllData.csv', index=False)
data = pd.read_csv('AllData.csv')
datacopy = data.copy()      # If data gets too modified we can use the datacopy method
data.shape     # 186850 x 6
# Using the string property and slicing the first two characters
data['Month'] = data['Order Date'].str[0:2]
data = data.dropna()
data.head()        # Generating the first three columns of the dataframe
temp = data[data['Order Date'].str[0:2] == 'Or']
temp.head(3)
data = data[data['Order Date'].str[0:2] != 'Or']
# Converting from string to integer data type
data['Month'] = data['Month'].astype('int32')
data.shape       # 185950 x 7
data['Price Each'] = pd.to_numeric(data['Price Each'])
data['Quantity Ordered'] = pd.to_numeric(data['Quantity Ordered'])
data['sales'] = data['Price Each'] * data['Quantity Ordered']
data.head(3)
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Month'] = pd.DatetimeIndex(data['Order Date']).month
data['Month'].shape      # 185950 x 1
month_file = data.groupby('Month')[['Price Each', 'Quantity Ordered', 'sales']].sum()
month_file.tail(3)
# We will first convert float values to integer values for both 'Price Each' and 'sales' column
month_file['Price Each'] = month_file['Price Each'].astype('int32')
month_file['sales'] = month_file['sales'].astype('int32')
month_file.tail(3)
# Now we divide Cost Revenue of month 'Price Each' by 100000 for visualization purposes
file1 = pd.DataFrame()
file1['Cost Revenue'] = month_file['Price Each'] / 10000  # In Ten Thousand
file1['Quantity'] = month_file['Quantity Ordered'] / 1000    # In Thousand
file1['Sales of Month'] = month_file['sales'] / 10000        # In Ten Thousand
file1 = file1.astype('int32')
file1.shape       # 12 x 3 (data of 12 months)
mon = np.arange(1, 13, 1)
sls = np.array(file1['Sales of Month'])
qty = np.array(file1['Quantity'])
cre = np.array(file1['Cost Revenue'])
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y = [182, 220, 280, 339, 315, 257, 264, 224, 209, 373, 319, 461]
plt.figure(figsize=(10, 6))
plt.bar(mon, sls, color='blue', alpha=1, width=0.5)
plt.xlabel('Months')
plt.xticks(mon)
for i in range(len(mon)):
    # i value to change x axis and y[i] value to change y axis
    plt.text(i+0.8, y[i], y[i], ha='left',bbox=dict(facecolor='red', alpha=0.8))
plt.ylabel('Sales in Ten Thousands')
plt.legend(['Monthly Sales'])
plt.title('Monthly Sales')
'''Thus the best Sales Month was of December and the Total Sales were 46,13,443'''
data['City'] = data['Purchase Address'].apply(lambda x: x.split(',')[1])      # The lambda data type creates a user defined function and performs it on the entire column sequence
data.head(3)
city_file = pd.DataFrame()
city_file['sales'] = data.groupby('City')['sales'].sum()
city_file['sales'] = city_file['sales'].astype('int32')
sls1 = np.array(city_file['sales'])
sls1 = sls1 / 100000
sls1 = sls1.astype('int32')
cty = np.array(['Atlanta', 'Austin', 'Boston', 'Dallas', 'Los Angeles', 'New York City', 'Portland', 'San Francisco', 'Seattle'])
plt.figure(figsize=(12, 8))
plt.bar(cty, sls1, color='darkblue')
plt.ylabel('Sales in Lakhs')
plt.xlabel('Cities')
plt.title("Sales In Each City Specifically", fontsize=24)
plt.show()
'''The Highest Sales were in the City San Francisco which were 82,62,203'''
data['Order Date'] = pd.to_datetime(data['Order Date'])
data.head(3)
data['Hour'] = pd.DatetimeIndex(data['Order Date']).hour
data['Minute'] = pd.DatetimeIndex(data['Order Date']).minute
data['Count'] = 1
data.shape
time = pd.DataFrame()
time = data[['Hour', 'Minute', 'Count', 'sales']]
time_hourly = pd.DataFrame()
time_hourly['Hour_time'] = time.groupby('Hour')['sales'].sum()
time_hourly['Hour_time'] = time_hourly['Hour_time'].astype('int32')
time_hourly.head(3)
time_hourly['Hour_time'] = time_hourly['Hour_time'] / 10000        # In Ten Thousand
time_hourly['Hour_time'] = time_hourly['Hour_time'].astype('int32')
hr = []
# The best hour for advertising will be when we have maximum sales for we will be having many customers active or online...
for i in range(1, 25):
    hr.append(i)
plt.plot(hr, time_hourly['Hour_time'], color='darkblue', marker='o', markerfacecolor='lightblue')
plt.xticks(hr)
plt.xlabel('Hours')
plt.ylabel('Sales in Ten Thousand')
plt.grid()
plt.title("Evaluating Best Advertisement Time", fontsize=24, color='red')
plt.show()
col = []
fcol = []
for i in range(1, 25):
    if i == 12 or i == 13 or i == 19 or i == 20 or i == 21:
        col.append('green')
        fcol.append('lightgreen')
    elif i == 3 or i == 4 or i == 5 or i == 6:
        col.append('red')
        fcol.append('orange')
    else:
        col.append('blue')
        fcol.append('lightblue')
# Passing colored values as list for the required colors
plt.scatter(hr, time_hourly['Hour_time'], color=col, facecolor=fcol)
plt.xticks(hr)
plt.xlabel('Hours')
plt.ylabel('Sales in Ten Thousand')
plt.title("Evaluated Best Advertisement Time", fontsize=20, color='red')
plt.grid()
plt.show()
'''Thus the best advertisement time is between 12:00 noon to 1:00 pm and from 7:00pm to 9:00 pm. Also the worst advertisement time is between 3:00 am to 6:00 am in morning.'''
items = pd.DataFrame()
items['Sales_Value'] = data.groupby('Product')['sales'].sum()
items.shape
items['Sales_Value'] = items['Sales_Value'].astype('int32')
items['Product'] = data['Product'].unique()
x = np.array(items['Product'].str[0:8])
print(x.size)
col1 = ['red', 'blue', 'green']
y = np.array(items['Sales_Value'])
plt.figure(figsize=(16, 8))
plt.bar(x, y, color=col1, alpha=0.4)
plt.grid()
for j in range(len(x)):
    plt.text(j, y[j]//2, y[j], ha='center', bbox=dict(facecolor='yellow', alpha=0.8))
plt.xticks(x, rotation=45)
plt.xlabel('Products Sold')
plt.ylabel('Sales of the Products (in Milllions)')
plt.title("Sales of the Products in an Year", fontsize=18, fontweight=100)
plt.show()
'''Thus the Best Product was of 20 Inches Monitor which gave the best Sales of 80,37,600 USD'''
#!/usr/bin/env python
# coding: utf-8

# ## PROBLEM SET 3

#  ### QUESTION 1

# In[188]:


#Step 1. Import the necessary libraries
import numpy as np
import pandas as pd


# In[189]:


#Step 2. Import the dataset
# Step 3. Assign it to a variable called users
users=pd.read_csv(r"https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user",sep="|")
users.head()


# In[190]:


#Step 4. Discover what is the mean age per occupation
mean_age_per_occup = users.groupby("occupation").age.mean()
print("Average age occupation wise")
print(mean_age_per_occup)


# In[191]:


#Step 5. Discover the Male ratio per occupation and sort it from the most to the least 

male_count=users.where(users.gender=="M").groupby(["occupation","gender"]).gender.count()
total_count=users.groupby("occupation").gender.count()
ratio=(male_count/total_count).sort_values(ascending=False)
print("Male ratio per occupation")
ratio


# In[192]:


#Step 6. For each occupation, calculate the minimum and maximum ages

users.groupby(["occupation"]).age.agg(["min","max"])


# In[193]:


#Step 7. For each combination of occupation and sex, calculate the mean age 

users.groupby(['occupation','gender']).agg({'age': ['mean']})


# In[194]:


#Step 8. For each occupation present the percentage of women and men

female_count=users.where(users.gender=='F').groupby(['occupation','gender']).gender.agg(['count'])
male_count=users.where(users.gender=='M').groupby(['occupation','gender']).gender.agg(['count'])
total_count=users.groupby('occupation').gender.agg(['count'])
male_ratio=(male_count/total_count)*100
female_ratio=(female_count/total_count)*100
pd.merge(male_ratio,female_ratio,on='occupation')


# ### Question2

# In[195]:


#Step 1. Import the necessary libraries

import numpy as np
import pandas as pd


# In[196]:


#Step 2. Import the dataset from this address
#Step 3. Assign it to a variable called euro12 
euro12=pd.read_csv(r"https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv",sep=",")
euro12.head()


# In[197]:


#Step 4. Select only the Goal column

euro12.Goals


# In[198]:


#Step 5. How many team participated in the Euro2012? 

print("Total teams participated in the Euro12 are :",len(euro12['Team'].unique()))


# In[199]:


#Step 6. What is the number of columns in the dataset?

print(" Total columns in the dataset are ", euro12.shape[1])


# In[200]:


#Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline
Data_Frame=euro12[["Team","Yellow Cards","Red Cards"]]
discipline=pd.DataFrame(Data_Frame)
discipline


# In[201]:


#Step 8. Sort the teams by Red Cards, then to Yellow Cards 

discipline.sort_values(by=['Red Cards', 'Yellow Cards'],ascending = True)


# In[202]:


#Step 9. Calculate the mean Yellow Cards given per Team
discipline.groupby("Team")["Yellow Cards"].mean()


# In[203]:


#Step 10. Filter teams that scored more than 6 goalsStep 11. Select the teams that start with G
euro12[euro12.Goals > 6]


# In[204]:


euro12[euro12["Team"].str.startswith("G")]


# In[205]:


#Step 12. Select the first 7 columns

euro12.iloc[:,:7]


# In[206]:


#Step 13. Select all columns except the last 3

euro12.iloc[:,:-3]


# In[207]:


# Step 14. Present only the Shooting Accuracy from England, Italy and Russia

euro12.loc[euro12.Team.isin(['England','Italy','Russia']),['Team','Shooting Accuracy']]


# ### Question3

# In[208]:


#Step 1. Import the necessary libraries

import numpy as np
import pandas as pd
import random


# In[209]:


# Step 2. Create 3 differents Series, each of length 100, as follows:
# • The first a random number from 1 to 4
# • The second a random number from 1 to 3
# • The third a random number from 10,000 to 30,000

x = pd.Series(np.random.randint(1,5,100))
y = pd.Series(np.random.randint(1,4,100))
z = pd.Series(np.random.randint(10000,30000,100))


# In[210]:


# Step 3. Create a DataFrame by joinning the Series by column

data = {'series1' : x,'series2':y,'series3':z}
df = pd.concat(data, axis = 1)
df


# In[211]:


# Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter

df.columns = ["bedrs","bathrs","price_sqr_meter"]
df


# In[212]:


# Step 5. Create a one column DataFrame with the values of the 3 Series and assign it to 'bigcolumn'
bigcolumn = pd.DataFrame(df['bedrs'].astype(str) + df['bathrs'].astype(str) + df['price_sqr_meter'].astype(str))
bigcolumn


# In[213]:


# Step 6. Ops it seems it is going only until index 99. Is it true? 

## Yes. we can see that the index is only going till 99 and the maximum rows are only 100
print(len(bigcolumn))


# In[214]:


# Step 7. Reindex the DataFrame so it goes from 0 to 299

bigcolumn.reindex(range(0, 300))


# ### Question 4

# In[215]:


# Step 1. Import the necessary libraries

import pandas as pd
from datetime import date
import datetime as dt 


# In[216]:


# Step 2. Import the dataset from the attached file wind.txt
# Step 3. Assign it to a variable called data and replace the first 3 columns by a proper datetime index
data= pd.read_fwf('wind.txt',parse_dates=[['Yr','Mo','Dy']])
data["Yr_Mo_Dy"] = pd.to_datetime(data["Yr_Mo_Dy"])
data.head()


# In[217]:


# Step 4. Year 2061? Do we really have data from this year? Create a function to fix it and apply it.
data["Yr_Mo_Dy"] = np.where(pd.DatetimeIndex(data["Yr_Mo_Dy"]).year < 2061,data.Yr_Mo_Dy,data.Yr_Mo_Dy - pd.offsets.DateOffset(years=100))
data.head()


# In[218]:


# Step 5. Set the right dates as the index. Pay attention at the data type, it should be datetime64[ns].
data["Yr_Mo_Dy"] = pd.to_datetime(data["Yr_Mo_Dy"])
display(data.dtypes)
data = data.set_index('Yr_Mo_Dy')
data.index
data


# In[219]:


# Step 6. Compute how many values are missing for each location over the entire record.They should be ignored in all calculations below
data.isnull().values.sum()


# In[220]:


new_data = data.dropna() 


# In[221]:


# Step 7. Compute how many non-missing values there are in total.

data.notnull().sum()


# In[222]:


# Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and all the times.
data.mean().mean()


# In[224]:


# Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean
#windspeeds and standard deviations of the windspeeds at each location over all the
#days A different set of numbers for each location
mini=new_data.min()
maxi=new_data.max()
mean=new_data.mean()
std=new_data.std()
var=[mini,maxi,mean,std]
index=["Min","Max","Mean","Std"]
loc_stats= pd.DataFrame(var,index)
loc_stats


# In[225]:


#Step 10. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day. A different set of numbers for each day.
day_stats = pd.concat([new_data.min(axis=1), new_data.max(axis=1), new_data.mean(axis=1), new_data.std(axis=1)], axis=1)

day_stats.rename(columns={0:'Min',1:'Max',2:'Mean',3:'Std'}, inplace=True)
day_stats


# In[226]:


# Step 11. Find the average windspeed in January for each location. Treat January 1961 and January 1962 both as January.
new_data[new_data.index.month==1].mean()


# In[227]:


# Step 12. Downsample the record to a yearly frequency for each location. 

downsample_data = data.resample('Y').ffill()
downsample_data


# In[228]:


#Step 13. Downsample the record to a monthly frequency for each location.
month_downsample= data.resample('M').ffill()
month_downsample.head()


# In[229]:


#Step 14. Downsample the record to a weekly frequency for each location.
week_downsample= data.resample('W').ffill()
week_downsample.head()


# In[230]:


# Step 15. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week (assume that the first week starts on January 2 1961) for the first 52 weeks.
week_stat=week_downsample.groupby(week_downsample.index.to_period('W')).agg(['min','max','mean','std'])
week_stat.loc[week_stat.index[1:53],:].head()


# ### Question5

# In[231]:


#Step1 - Importing necessary libraries : 
import pandas as pd
import numpy as np


# In[232]:


#Step2 & Step3 - Importing dataset from git-hub link and assigning it a variable called chipo :
link = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_table(link)


# In[233]:


#Step 4. See the first 10 entries
chipo.head(10)


# In[234]:


# Step 5. What is the number of observations in the dataset? 

print("Total Observations ={} *{} ".format(chipo.shape[0],chipo.shape[1]))


# In[235]:


# Step 6. What is the number of columns in the dataset?

len(chipo.columns)


# In[236]:


# Step 7. Print the name of all the columns. 

chipo.columns


# In[237]:


# Step 8. How is the dataset indexed?

chipo.index


# In[238]:


# Step 9. Which was the most-ordered item?

most_ordered=chipo.groupby('item_name').sum()
most_ordered=most_ordered.sort_values(by=['quantity'], ascending = False)
most_ordered.head(1)


# In[239]:


# Step 10. For the most-ordered item, how many items were ordered?

choice_ordered=chipo.groupby('choice_description').sum()
choice_ordered=choice_ordered.sort_values(by=['quantity'], ascending = False)
choice_ordered.head(5)


# In[240]:


# Step 12. How many items were orderd in total?

chipo.groupby('quantity').quantity.sum().sum()


# In[241]:


# Step 13.
#• Turn the item price into a float
#• Check the item price type
#• Create a lambda function and change the type of item price
#• Check the item price type
chipo.item_price.dtype


# In[242]:


try:                                                 
    convertToFloat = lambda x: float(x[1:-1])
    chipo.item_price = chipo.item_price.apply(convertToFloat)
except:TypeError    


# In[243]:


chipo.item_price.dtype


# In[244]:


#Step 14. How much was the revenue for the period in the dataset? 

revenue = (chipo['quantity'] * chipo['item_price'])
revenue.sum()


# In[245]:


# Step 15. How many orders were made in the period?

chipo.order_id.value_counts().count()


# In[246]:


# Step 16. What is the average revenue amount per order? 

avg_revenue = chipo['quantity'] * chipo['item_price']
d = order_grouped = chipo.groupby(by=['order_id']).sum()
order_grouped.mean()


# In[247]:


# Step 17. How many different items are sold?

chipo.item_name.value_counts().count()


# ### Question 6

# In[248]:


# Create a line plot showing the number of marriages and divorces per capita in the
#U.S. between 1867 and 2014. Label both lines and show the legend.
#Don't forget to label your axes!

import matplotlib.pyplot as plt
import pandas as pd


# In[249]:


marriage_data = pd.read_csv('us-marriages-divorces-1867-2014.csv')
marriage_data.head()


# In[250]:


year=marriage_data.Year.values
marriage=marriage_data.Marriages_per_1000.values
divorce=marriage_data.Divorces_per_1000.values

plt.plot(year,marriage,color="Blue")
plt.plot(year,divorce,color="Yellow")
plt.xlabel("Years")
plt.ylabel("Marriages vs Divoces")
plt.title("The number of marriages and divorces per capita in the U.S. between 1867 and 2014.")
plt.show()


# ### Question 7

# In[251]:


# Create a vertical bar chart comparing the number of marriages and divorces per
#capita in the U.S. between 1900, 1950, and 2000.
#Don't forget to label your axes!
years=[1900,1950,2000]
data = marriage_data.loc[marriage_data['Year'].isin(years)]
plt.bar(data['Year'],data['Marriages_per_1000'],color="Green")
plt.bar(data['Year'],data['Divorces_per_1000'],color="Blue")
plt.title("Total number of Marraiges and Divorces per capita in US between 1900 to 2000 \n\n\n")
plt.xlabel("Between 1900 and 2000")
plt.ylabel("Marriages & Divorces per capita in US")
plt.show()


# ### Question8

# In[252]:


# Create a horizontal bar chart that compares the deadliest actors in Hollywood. Sort
#the actors by their kill count and label each bar with the corresponding actor's name.
#Don't forget to label your axes!

actors = pd.read_csv('actor_kill_counts.csv')
actors.head(10)


# In[253]:


sort_actors =actors.sort_values(by='Count',ascending=True)
actor_Names= actors.Actor
plt.barh(sort_actors['Actor'],sort_actors['Count'],color="blue")
plt.xlabel("Number of Kills")
plt.ylabel("Actor Name")
plt.title("The deadliest actors in Hollywood \n\n ")
plt.show()


# ### Question9

# In[254]:


#Create a pie chart showing the fraction of all Roman Emperors that were
#assassinated.

roman_data= pd.read_csv('roman-emperor-reigns.csv')
roman_data.head()


# In[255]:


lowerData= roman_data.where(roman_data.Cause_of_Death=="Assassinated").Cause_of_Death.count()
remaining= roman_data.Cause_of_Death.count()-lowerData
label=["Other Cause of Deaths","Assassinated"]
plt.pie([remaining,lowerData],labels=label,autopct='%.2f%%')
plt.title("Roman Emperors")
plt.show()


# ### Question 10

# In[256]:


#Create a scatter plot showing the relationship between the total revenue earned by
#arcades and the number of Computer Science PhDs awarded in the U.S. between
#2000 and 2009.
import seaborn as sns


# In[257]:


revenue_data = pd.read_csv('arcade-revenue-vs-cs-doctorates.csv')
revenue_data.head(10)


# In[258]:


sns.scatterplot(x=revenue_data['Total Arcade Revenue (billions)'], y=revenue_data['Computer Science Doctorates Awarded (US)'], data= revenue_data, hue = 'Year')


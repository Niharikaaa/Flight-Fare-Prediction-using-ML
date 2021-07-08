#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #data visualisation library (used to make well defined graphs)


# In[2]:


#import the dataset
train_data = pd.read_excel(r"C:\Users\Niharika\OneDrive\Desktop\Data_Train.xlsx")


# In[3]:


pd.set_option('display.max_columns' , None) #to display all the columns we put the max column value as none


# In[4]:


train_data.head() #head displays top 5 values of the dataset


# In[5]:


train_data.info()              #as data type of date and time is also object and might give some problems in prediction so we change it to datetime type


# In[6]:


train_data["Duration"].value_counts()           #to count how many values of a particular type are there in the duration column


# In[7]:


train_data.isnull().sum()        #isnull().sum() retuns if there are any null values in the dataset and what are those values and isnull() returns if there are some null values or not as true or false only


# In[8]:


train_data.dropna(inplace = True) #dropna will remove the null values from dataset and inplace = true means modify the same dataset and not create a new one


# In[9]:


train_data.isnull().sum()


# In[10]:


# as data of journey is in type object so we have to convert it to tmestamp i.e. intenger value for proper prediction
#dt.day will give us only the day of the date
#dt.month will give us the month of the date
#then will we remove date_of_journey column and the new column will be journey_day representing the day of the date
train_data["Journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day


# In[11]:


train_data["Journey_month"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.month


# In[12]:


train_data.head()


# In[13]:


train_data.drop(["Date_of_Journey"] , axis=1, inplace=True)


# In[14]:


# now we have to convert depature time in hrs and mins respectively
train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour
train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute
train_data.drop(["Dep_Time"] , axis=1, inplace=True)


# In[15]:


train_data.head()


# In[16]:


# now we have to convert depature time in hrs and mins respectively
train_data["Arrival_hour"] = pd.to_datetime(train_data["Arrival_Time"]).dt.hour
train_data["Arrival_min"] = pd.to_datetime(train_data["Arrival_Time"]).dt.minute
train_data.drop(["Arrival_Time"] , axis=1, inplace=True)


# In[17]:


train_data.head()


# In[18]:


#duration is difference between arrival time and depature time
#suppose it is 2hr 50 mins then we will split it in middle with one part being 2hr and other as 50 mins
#then check if not 2 parts and if h in there then we add 0mins else add 0hr
duration=list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split())!=2:
        if "h" in duration[i]:
            duration[i]=duration[i].strip()+" 0m"
        else:
            duration[i]="0h "+duration[i]
duration_hours=[]
duration_mins=[]
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))
    
    


# In[19]:


train_data["Duration_hours"]=duration_hours
train_data["Duration_mins"]=duration_mins


# In[20]:


train_data.drop("Duration",axis=1,inplace=True)


# In[21]:


train_data.head()


# Handling Categorical data:
# 1) Ordinal: data are in order, use label encoder
# 2) nominal: data are not in order, use one hot encoding
# 
# Now as shown in graph, the data in not in order as jet airways price is higher than other so we will use nominal that is one hot encoding. Then we will create dummy variables.

# In[22]:


train_data["Airline"].value_counts()
#plotting airline against price using cat plot

sns.catplot(y="Price",x="Airline",data=train_data.sort_values("Price",ascending=False),kind="boxen",height=6,aspect=3)


# In[23]:


#one hot encoding of Airline (Nominal data)
Airline = train_data[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first=True)
Airline.head()


# In[24]:


train_data["Source"].value_counts()


# In[25]:


# Source vs Price

sns.catplot(y = "Price", x = "Source", data = train_data.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)


# In[26]:


source = train_data[["Source"]]

source = pd.get_dummies(source, drop_first= True)
source.head()


# In[27]:


train_data["Destination"].value_counts()


# In[28]:


destination = train_data[["Destination"]]

destination = pd.get_dummies(destination, drop_first= True)
destination.head()


# In[29]:


train_data["Route"]


# In[30]:


train_data.drop("Route",axis=1,inplace=True)


# In[31]:


train_data.drop("Additional_Info",axis=1,inplace=True)


# In[32]:


train_data["Total_Stops"].value_counts()


# In[33]:


#as ordinal categorical data so use label encoder and replace non stop as 0 , 1 stop as 1 and so on
train_data.replace({"non-stop": 0, "1 stop": 1 , "2 stops": 2, "3 stops": 3, "4 stops": 4} , inplace= True)


# In[34]:


train_data.head()


# In[35]:


#concatenate train dataframe with one-hot encoded data
data_train = pd.concat([train_data, Airline , source , destination], axis=1)


# In[36]:


data_train.head()


# In[37]:


data_train.drop(["Airline","Source","Destination"], axis=1, inplace=True)


# In[38]:


data_train.head()


#  now in the dataset we have all integer values only

# In[39]:


data_train.shape #now we have 30 columns


# ## Test Set

# In[40]:


test_data = pd.read_excel(r"C:\Users\Niharika\OneDrive\Desktop\Test_set.xlsx")


# In[41]:


test_data.head()


# In[42]:


test_data.info()


# In[43]:


test_data.isnull().sum()


# In[44]:


test_data.dropna(inplace=True)


# In[45]:


# Date_of_Journey
test_data["day_of_journey"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["month_of_journey"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_Hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_Min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# In[46]:


airlines=test_data[["Airline"]]
airlines=pd.get_dummies(airlines,drop_first=True)


source = test_data[["Source"]]

source = pd.get_dummies(source, drop_first= True)
destination = test_data[["Destination"]]

destination = pd.get_dummies(destination, drop_first = True)

destination.head()


# In[47]:


#as ordinal categorical data so use label encoder and replace non stop as 0 , 1 stop as 1 and so on
test_data.replace({"non-stop": 0, "1 stop": 1 , "2 stops": 2, "3 stops": 3, "4 stops": 4} , inplace= True)


# In[48]:


test_data.head()


# In[49]:


test_data=pd.concat([test_data,airlines,destination,source],axis=1)


# In[50]:


test_data.drop(["Airline","Source","Destination","Route","Total_Stops"],axis=1,inplace=True)


# In[51]:


test_data.head()


# In[52]:


test_data.shape


# In[53]:


data_train.columns


# In[54]:


import matplotlib.pyplot as plt
plt.figure(figsize = (18,18))
sns.heatmap(train_data.corr(), annot = True, cmap = "RdYlGn")

plt.show()


# In[72]:


X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]


# In[74]:


X.head()


# In[75]:


y = data_train.iloc[:, 1]


# In[77]:


y.head()


# In[78]:


from sklearn.ensemble import ExtraTreesRegressor


# In[79]:


selection = ExtraTreesRegressor()


# In[80]:


selection.fit(X, y)


# In[81]:


print(selection.feature_importances_)


# In[82]:


plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh' , color ='lightblue')
plt.show()


# as total_stops has highest value so it is most imp

# In[83]:


from sklearn.model_selection import train_test_split


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[85]:


from sklearn.ensemble import RandomForestRegressor


# In[86]:


reg_rf = RandomForestRegressor()


# In[87]:


reg_rf.fit(X_train, y_train)


# In[88]:


y_pred = reg_rf.predict(X_test)


# In[89]:


reg_rf.score(X_train, y_train)


# In[90]:


reg_rf.score(X_test, y_test)


# In[91]:


sns.distplot(y_test - y_pred , color ='red')
plt.show()


# In[92]:


plt.scatter(y_test, y_pred, alpha = 0.5 , color ='lightblue')
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[93]:


from sklearn import metrics


# In[94]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[95]:


#RMSE/(max(DV)-min(DV))      used to normalise the values
2090.5509/(max(y)-min(y))


# In[96]:


metrics.r2_score(y_test, y_pred)


# Hyperparameter tuning is the process of determining the right combination of hyperparameters that allows the model to maximize model performance. Setting the correct combination of hyperparameters is the only way to extract the maximum performance out of models.
# Hyperparameter tuning methods 
# In this section, I will introduce all of the hyperparameter tuning methods that are popular today. 
# 
# Random Search
# In the random search method, we create a grid of possible values for hyperparameters. Each iteration tries a random combination of hyperparameters from this grid, records the performance, and lastly returns the combination of hyperparameters which provided the best performance.
# 
# Grid Search
# In the grid search method, we create a grid of possible values for hyperparameters. Each iteration tries a combination of hyperparameters in a specific order. It fits the model on each and every combination of hyperparameter possible and records the model performance. Finally, it returns the best model with the best hyperparameters.

# In[97]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]


# In[98]:


max_features = ['auto', 'sqrt']


# In[99]:


max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]


# In[100]:


min_samples_split = [2, 5, 10, 15, 100]


# In[101]:


min_samples_leaf = [1, 2, 5, 10]


# In[102]:


random_grid = {'n_estimators': n_estimators,'max_features': max_features,'max_depth': max_depth,'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[103]:


from sklearn.model_selection import RandomizedSearchCV


# In[104]:


rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[105]:


rf_random.fit(X_train,y_train)


# In[106]:


rf_random.best_params_


# In[107]:


prediction = rf_random.predict(X_test)


# In[108]:


plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction , color='red')
plt.show()


# In[109]:


plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5 , color ='lightblue')
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[110]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# ###### pandas - used to perform data manipulation and analysis
# 
# ###### numpy - used to perform a wide variety of mathematical operations on arrays
# 
# ###### matplotlib - used for data visualization and graphical plotting
# 
# ###### seaborn - built on top of matplotlib with similar functionalities
# 
# ###### %matplotlib - to enable the inline plotting.
# 
# ###### warnings - to manipulate warnings details 
# 
# ###### filterwarnings('ignore') is to ignore the warnings thrown by the modules (gives clean results)

# ## Step-1: Loading the Dataset

# In[2]:


bike_info=pd.read_csv("day.csv")
bike_info.head(10)


# In[3]:


bike_info = bike_info.rename(columns={'weathersit':'weather',
                                       'yr':'year',
                                       'mnth':'month',
                                       'hr':'hour',
                                       'hum':'humidity',
                                       'cnt':'count'})
bike_info.head()


# In[4]:


# checking for the null values in column data
bike_info.isnull().sum() #ZERO NULL VALUES


# #dropping the unnecessary column "casual" and "registered".
# 
# #To use feature engineering, encoding categorical column.

# In[5]:


# statistical info
bike_info.describe()


# #There are no missing values in the dataset.

# In[6]:


# datatype info
bike_info.info()


# #The datatype of the remaining column is float and integer.
# 
# 
# #As we see all the components of date like month, day, year  are already present in the dataset hence we can drop unnecessary columns 'dteday' also instant represent index does not seem useful here.
# 
# #casual and registered seems to be the breakup by category for count column.

# In[7]:


bike_info.drop(columns=['instant', 'dteday','casual','registered'], inplace = True)
bike_info.head()


# In[8]:


# Encoding/mapping the season column

bike_info.season = bike_info.season.map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})
# Encoding/mapping the month column

bike_info.month = bike_info.month.map({1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'june',7:'july',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'})
# Encoding/mapping the weekday column

bike_info.weekday = bike_info.weekday.map({0:'sun',1:'mon',2:'tue',3:'wed',4:'thu',5:'fri',6:'sat'})
# Encoding/mapping the weathersit column

bike_info.weather = bike_info.weather.map({1:'Clear',2:'Misty',3:'Light_snowrain',4:'Heavy_snowrain'})


# In[9]:


bike_info.head()


# In[10]:


# check for null values
bike_info.shape


# ## Step-2: Visualizing the Data

# Data Visualisation using matplotlib and seaborn libraries.

# ##### Visualising Numeric Variables.
# Let's make a pairplot of all the numeric variables

# In[11]:


# Analysing/visualizing the numerical columns
sns.pairplot(data=bike_info,vars=['temp','atemp','humidity','windspeed','count'])
plt.show()


# In[12]:


# Checking the correlation between the numerical variables

plt.figure(figsize = (8,4))
matrix = np.triu(bike_info[['temp','atemp','humidity','windspeed','count']].corr())
sns.heatmap(bike_info[['temp','atemp','humidity','windspeed','count']].corr(), annot = True)
plt.title("Correlation b/w Numerical Variables")
plt.show()


# From the above scatter plot and the heat map, we can notice that, there is linear relationship between temp and atemp(has high co-relation with value 0.99). Both of the parameters cannot be used in the model due to multicolinearity.
# 

# - By looking at the pair plot & heat map, temp & atemp variables have the highest (0.63) correlation with target variable 'count'.

# ##### Visualising Categorical Variables
# There are few categorical variables as well. Let's make a boxplot for some of these variables.

# In[13]:


# Analysing/visualizing the categorical columns
# to see how predictor variable stands against the target variable

plt.figure(figsize=(20, 12))
plt.subplot(2,4,1)
sns.boxplot(x = 'season', y = 'count', data = bike_info)
plt.subplot(2,4,2)
sns.boxplot(x = 'month', y = 'count', data = bike_info)
plt.subplot(2,4,3)
sns.boxplot(x = 'weekday', y = 'count', data = bike_info)
plt.subplot(2,4,4)
sns.boxplot(x = 'weather', y = 'count', data = bike_info)
plt.subplot(2,4,5)
sns.boxplot(x = 'holiday', y = 'count', data = bike_info)
plt.subplot(2,4,6)
sns.boxplot(x = 'workingday', y = 'count', data = bike_info)
plt.subplot(2,4,7)
sns.boxplot(x = 'year', y = 'count', data = bike_info)
plt.show()


# In[14]:


# Using function to create barplot related to categorical columns

def plot_cat(column):
    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    sns.barplot(column,'count',data=bike_info)
    plt.subplot(1,2,2)
    sns.barplot(column,'count',data=bike_info, hue='year')
    plt.legend(labels=['2018', '2019'])
    plt.show()


# In[15]:


#visualizing season column
plot_cat('season')


# Fall season seems to have attracted more booking. And, in each season the booking count has increased drastically from 2018 to 2019.

# In[16]:


#visualizing month column
plot_cat('month')


# Most of the bookings has been done during the month of May, June, July, August, September and October. Trend increased starting of the year tillmid of the year and then it started decreasing as we approached the end of year. Number of booking for each month seems to have a good increase from 2018 to 2019.

# In[17]:


#visualizing weather column
plot_cat('weather')


# We can say by looking at above plots Clear weather attracts more booking which is quite obvious and in comparison to previous year, i.e 2018, booking increased in 2019 for all weather states.

# In[18]:


#visualizing weekday column
plot_cat('weekday')


# By looking at the above plots we can say Days Sat, Sun & Thu, Fri have more bookings and even Wednesday has good number of bookings incomparison to Mondays & Tuesdays.
# 
# Now, lets have a look at holidays.

# In[19]:


#visualizing holiday column
plot_cat('holiday')


# In[20]:


#visualizing workingday column
plot_cat('workingday')


# Does not seem much difference on number of bookings whether its a working day or non-working day, but the count increased from 2018 to 2019.

# In[21]:


#visualizing year column
plot_cat('year')


# As we have noticed in all other visualizations as well, year 2019 has more number of bookings in compariison to year 2018. 
# 
# Which also shows that the business is growing.

# ## Step 3: Data Preparation

# - In order to fit a regression line, we would need numerical values and not string.
# 
# - Some variables such as 'months', 'weekday' has multiple levels. We need to convert these levels into integer as well.
# 
# - For this, we will use something called 'dummy variables'.

# ###### Step 3.1: Dummy Variable Creation

# In[22]:


# Dummy variable creation for month, weekday, weathersit and season variables.

months_df=pd.get_dummies(bike_info.month, drop_first=True)
                             
weekdays_df=pd.get_dummies(bike_info.weekday, drop_first=True)
                               
weathersit_df=pd.get_dummies(bike_info.weather, drop_first=True)
                                
seasons_df=pd.get_dummies(bike_info.season, drop_first=True)


# In[23]:


# Let's Merge the dataframe, with the dummy variable dataset. 
bike_info = pd.concat([bike_info, 
                       months_df,
                       weekdays_df,
                       weathersit_df,
                       seasons_df],
                       axis=1)


# In[24]:


#Displaying merged Dataframe
bike_info.head()


# In[25]:


bike_info.info()


# In[26]:


# dropping unnecessary columns as we have already created dummy variable out of it.

bike_info.drop(['season','month','weekday','weather'], axis = 1, inplace = True)


# In[27]:


# check the head of dataframe now
bike_info.head()


# In[28]:


bike_info.shape


# ## Step 4: Splitting the Data into Training and Testing Sets¶
#     
# As we know, the first basic step for regression is performing a train-test split.

# In[29]:


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively.
np.random.seed(0)
df_train, df_test = train_test_split(bike_info, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[30]:


# check the shape of training datatset
df_train.shape


# In[31]:


# check the shape of testing datatset
df_test.shape


# ### Rescaling the Features

# If we don't have comparable scales, then some of the coefficients as obtained by fitting the regression model might be very large or very small as compared to the other coefficients. This might become very annoying at the time of model evaluation. So it is advised to use standardization or normalization so that the units of the coefficients obtained are all on the same scale. As we know, there are two common ways of rescaling:
# 
# 1. Min-Max scaling
# 2. Standardisation (mean-0, sigma-1)
# 
# Let's use Min-Max Scaling this time.

# In[32]:


# Using MinMaxScaler to Rescaling the features
# Importing MinMaxScaler from sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler


# In[33]:


# 1. instantiate an object
scaler = MinMaxScaler()

#Creating a list of numeric variables
num_vars = ['temp','atemp','humidity','windspeed','count']

#2. Fit on Data
# Apply scaler() to all the columns except the 'binary' and 'dummy' variables
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[34]:


# verifying the head after appying scaling.
df_train.head()


# In[35]:


# describing the dataset
df_train.describe()


# ## Step 5: Training the Model

# In[36]:


# Checking the correlation coefficients to see which variables are highly correlated
plt.figure(figsize = (25,25))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


#  As per above heatmap count seems to have correlations with 'year' and 'temp'(with 0.59 value).Similarly, 'Misty' and 'humidity' have some correlation(with 0.48 value). 'Spring' season with 'Jan' and 'Feb' month, 'Summer' season with 'may' month and 'Winter' season with 'oct' and 'nov' month show good correlation.

# In[37]:


#Let's visualize some of the correlations using pair plot.
sns.pairplot(data=bike_info,vars=['temp','atemp','humidity','windspeed','count'])
plt.show()


# - Above plot shows "count" & "temp/atemp" are having some Linear Relationship
# - So, we pick 'temp' as the first variable and we'll try to fit a regression line to that.

# In[38]:


#checking on head of the df_train before we divide the model
df_train.head()


# ### Dividing into X and Y sets for the model building

# In[39]:


y_train = df_train.pop("count")
x_train = df_train


# In[40]:


x_train


# In[41]:


y_train


# ## Step 6: Building a linear model

# - We will be using the **LinearRegression function from SciKit Learn** for its compatibility with RFE (which is a utility from sklearn)

# ##### RFE
# Recursive feature elimination

# In[42]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[43]:


# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(x_train, y_train)

rfe = RFE(estimator=lm, n_features_to_select=15)            # running RFE
rfe = rfe.fit(x_train, y_train)


# In[44]:


rfe


# In[45]:


#List of variables selected in top 15 list
list(zip(x_train.columns,rfe.support_,rfe.ranking_))


# In[46]:


# selecting the selected variable via RFE in col list
col = x_train.columns[rfe.support_]
col


# In[47]:


# checking which columns has been rejected
x_train.columns[~rfe.support_]


# In[48]:


# dataframe with RFE selected variables
x_train_rfe = x_train[col]


# ###### we have high VIF for humidity.

# ### Step 6.1: Building model using statsmodel, for the detailed statistics

# In[49]:


#Building the Model
# Adding a constant variable 
import statsmodels.api as sm
x_train_lm_1 = sm.add_constant(x_train_rfe)


# In[50]:


x_train_lm_1


# In[51]:


x_train_rfe


# In[52]:


# Running the linear model
lr_1 = sm.OLS(y_train,x_train_lm_1).fit()


# In[53]:


#Let's see the summary of our linear model
print(lr_1.summary())


# - The 'p-value' for each term tests the null hypothesis that the coefficient is equal to zero (no effect). A low p-value (< 0.05) indicates that you can reject the null hypothesis. Predictor that has a low p-value is likely to be a meaningful addition to th model. So, we can drop variables having p value > 0.05.

# - A 'variance inflation factor (VIF)' is a measure of the amount of multicollinearity in regression analysis. Multicollinearity exists when there is a correlation between multiple independent variables in a multiple regression model. This can adversely affect the regression results.

# ###### So p-value does not seem high for these variables, hence let's have a look at 'VIF'  ('variance inflation factor').

# ### Checking VIF

# In[54]:


# importing variance_inflation_factor from statsmodels.stats.outliers_influence
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[55]:


# Check for the VIF values of the feature variables. 
# Generic function to calculate VIF of variables

def calculateVIF(df):
    vif = pd.DataFrame()
    vif['Features'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif 


# In[56]:


# checking VIF for x_train_rfe, as its the same model without constant instead removing the constant here.
calculateVIF(x_train_rfe)


# ###### VIF value for 'humidity' - 30.94 (>5) & 'temp' - 17.80 (>5) is quite high at this point, we can drop column 'humidity' first and check the VIF again.

# ### Dropping the variable and updating the model

# In[57]:


# Dropping 'humidity' column to improve the model
X_train_new = x_train_rfe.drop(['humidity'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# ######  VIF values seems to be better now after dropping 'humidity' but 'temp' - 5.17 is still greater than 5, but let's have a look at other statistical values of new model.

# In[58]:


# Building 2nd linear regression model
# Adding a constant variable
X_train_lm_2 = sm.add_constant(X_train_new)


# In[59]:


# Running the linear model
lr_2 = sm.OLS(y_train,X_train_lm_2).fit() 


# In[60]:


#Let's see the summary of our linear model
print(lr_2.summary())


# ###### p-value of 'summer' seems bit high, let's drop this column and check the VIF.

# ### Dropping the variable and updating the model

# In[61]:


# We can drop summer variable as it has bit high p-value
X_train_new = X_train_new.drop(['summer'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# ######  VIF values seems to be better now after dropping 'summer', but lets have a look at other statistical values of new model.

# In[62]:


# Building 3rd linear regression model
# Adding a constant variable
X_train_lm_3 = sm.add_constant(X_train_new)


# In[63]:


# Running the linear model
lr_3 = sm.OLS(y_train,X_train_lm_3).fit()


# In[64]:


#Let's see the summary of our linear model
print(lr_3.summary())


# ###### p-values for all the predictors seems to be significant.(<0.05)

# In[65]:


# Run the function to calculate VIF of final model
calculateVIF(X_train_new)


# ###### We can consider the this model i.e lr_5, as it seems to have very low multicolinearity between the predictors and the p-values for all the predictors seems to be significant. F-Statistics value of 197.9 (which is greater than 1)  and p-value are less than 0.05 above, also VIF's are less than 5) which states that the overall model is significant.

# ## Step:7 Residual Analysis of the train data and validation 

# In[66]:


#Let's do Residual Analysis
X_train_lm_3


# In[67]:


# calculating y_train_pred
y_train_pred = lr_3.predict(X_train_lm_3)


# In[68]:


# calculating residual & visulalizing it using distplot
res = y_train_pred - y_train


# ###### Normality of error terms

# In[69]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 25) 
fig.suptitle('Error Terms', fontsize = 21)          # Plot heading 
plt.xlabel('Errors', fontsize = 15)                 # X-label


# ###### By looking at above distplot we can say that 'Error terms' are following 'Normal Distribution'.

# # Step 8: Making Predictions

# ### Making Predictions Using the Final Model
# 
# Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make predictions using the final, i.e. 3rd model(lr_3).

# In[70]:


# Applying the scaling on the test sets
num_vars = ['temp', 'atemp', 'humidity', 'windspeed','count']
df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()


# In[71]:


df_test.describe()


# #### Dividing into X_test and y_test

# In[72]:


y_test = df_test.pop('count')
x_test = df_test


# In[78]:


col1 = X_train_new.columns

x_test = x_test[col1]

# Adding constant variable to test dataframe
X_test_lm_3 = sm.add_constant(x_test)


# In[83]:


# Making predictions
y_pred = lr_3.predict(X_test_lm_3)


# ## Step 9: Model Evaluation
# 
# Let's now plot the graph for actual versus predicted values.

# In[88]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 18)   # Plot heading 
plt.xlabel('y_test', fontsize = 15)               # x-label
plt.ylabel('y_pred', fontsize = 15)               # y-label


# ### Step 9.1:  Calculating R^2 value for the test dataset

# In[101]:


# Calculating R^2 value for the test dataset
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
r2


# ###  Step 9.2:  Calculating Adjusted-R^2 value for the test dataset

# In[95]:


# Calculating Adjusted-R^2 value for the test dataset
adjusted_r2 = round(1-(1-r2)*(x_test.shape[0]-1)/(x_test.shape[0]-x_test.shape[1]-1),4)
print(adjusted_r2)


# In[102]:


#checking parameters of our Model
round(lr_3.params,4)


# ###  Step 9.3: We can see that the equation of our best fitted line is:

# ## count= 0.2893 + 0.2348 * year + (-0.0913) * holiday + 0.4026 * temp + (-0.1540) * windspeed + (-0.0510) * dec + (-0.0556) * jan + (-0.0643) * july + (-0.0488) * nov + 0.0537 * sep + (-0.2949) * Light_snowrain + (-0.0812) * Misty + (-0.1034) * spring + 0.0650 * winter

# In[109]:


### Visualizing the fit on the test data
# plotting a Regression plot

plt.figure()
sns.regplot(x=y_test, y=y_pred, ci=68, fit_reg=True,scatter_kws={"color": "green"}, line_kws={"color": "red"})
plt.title('y_test vs y_pred', fontsize=20)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)
plt.show()


# ##  Step 10: Comparision between Training and Testing dataset:

# - Train dataset R^2          : 0.838
# - Test dataset R^2           : 0.816
# - Train dataset Adjusted R^2 : 0.834
# - Test dataset Adjusted R^2  : 0.804

# ## Step 11: Conclusion

# ###### As per our final Model, the top predictor components that influences the bike booking are:
# 
# - Temperature
# - Weather
# - Year
# - windspeed
# 
# So, it's suggested to consider these components utmost importance while planning, to achive maximum Booking.

# #### The variables which are affecting the count variable are as follows:
# 1. year 
# 2. holiday
# 3. temp 
# 4. windspeed
# 5. dec
# 6. jan 
# 7. july
# 8. nov 
# 9. sep 
# 10. Light_snowrain 
# 11. Misty
# 12. spring 
# 13. winter
# 

# ### Hence, Demand of bikes depend on above listed variables.
# Error terms are unequally distributed and does not follow any pattern, as there is no curve in the plot. It indicates that it is just the white noise.

# ## Thank you!

# .

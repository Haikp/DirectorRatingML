# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# %%
data = pd.read_csv('movies.csv', header=None)
data.drop([0], inplace=True)
# we will try to estimate the gross income made by the movie
cols = {0:'Name', 1:'Rating', 2:'Genre', 3:'Year', 4:'Released', 5:'Score', 6:'Votes',7:'Directed' ,8:'Writer' ,9:'Star' ,10:'Country', 11: 'Budget', 12: 'Gross', 13:'Company', 14:'Runtime'}

#data that has dropped NA values 
# est_data = data.dropna(subset=[0,1,2,3,4,5,6,7,8,9,10,11,13,14])
est_data = data.dropna(subset=[5,6,7,11])
est_data.rename(columns=cols, inplace=True)
# data.drop(data.columns[[0,1,2,4,7,8,9,10,13,14]], axis=1, inplace=True)
data.drop(data.columns[[0,1,2,3,4,5,6,7,8,9,10,13,14]], axis=1, inplace=True)

# cols = {3:'Year', 5:'Rating', 6:'Number of votes', 11: 'Budget', 12: 'Gross'}
cols = {11: 'Budget', 12: 'Gross'}
data.rename(columns=cols, inplace=True)
data = data.astype(np.float64)

scaler = StandardScaler()

# %%
#data.fillna(data.mean(axis=0), inplace=True)
gross_est = data
gross_est.dropna(axis=0, how='any', inplace=True)
#gross_est = np.array(data)
X = data.iloc[:, 0]
Y = data.iloc[:, 1]


# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# %%
# we will be predicting the gross with the budget alone
  # after observing the data, the other independent variables provide
    # convoluted data when assessing how much the movie made
      # Name: Irrelevant data
      # Rating: terrible movies can profit regardless, ex Morbius
      # genre: Irrelevant data, not ordinal data
      # year: when the movie came out should not be a factor
        # could be useful when comparing when a genre was good
      # released: same as year above
      # votes: refers to number of votes, how many people watched a
        # movie does not indicate how good it was, ex Morbius
      # Director: May be useful, however not necessary when determining gross
        # arguably able to determine budget instead
      # Writer: same as director
      # Star: Same as director
      # Country: location should not matter
      # Company: not all companies on this list are able to determine gross
        # and budget by name alone, smaller companies would be difficult
      # Runtime: duration of movie should not matter
x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=None)

#print(x_train, y_train)
#normalization
x_train = x_train - x_train.min()
x_train = x_train / x_train.max()
x_test = x_test - x_test.min()
x_test = x_test / x_test.max()
y_train = y_train - y_train.min()
y_train = y_train / y_train.max()
y_test = y_test - y_test.min()
y_test = y_test / y_test.max()

reg = LinearRegression().fit(x_train.values.reshape(-1, 1), y_train)
pred = reg.predict(x_test.values.reshape(-1, 1))

plt.scatter(x_train,y_train)
plt.plot(x_test, pred, color='k')
plt.axline((0, 0), slope=1, color = 'r')
plt.show()

pd.DataFrame(pred)

# %%
RMSE = np.sqrt(mean_squared_error(y_test, pred))
MAE = mean_absolute_error(y_test, pred)
# X_train
print(RMSE)
print(MAE)

#values are close to be decent in predicting accurate values

# %% [markdown]
# ```
# pseudo code
# grab original data with everything in it
# separate data in to 2 data sets
# drop datasets that contain null in anywhere expect gross
# of the remaining data, separate into:
#   1: data that has gross as null
#   2: the rest of the data that does not have gross as null
# use the linear model to predict the gross
# put both data sets back together
# that will be our new dataset we are working with
# refer to the google doc with our proposal on it for further instructions
# 

# %%
#formatting 
pd.options.display.float_format = '{:,}'.format

#split into gross filled or not
na_df = est_data[est_data['Gross'].isnull()]
filled_df = est_data[est_data['Gross'].notnull()]

#drop columns 
na_df_copy = na_df.copy()
new_na_df = na_df.drop(columns=['Name', 'Rating', 'Genre', 'Year', 'Released', 'Score', 'Votes', 'Directed', 'Writer', 'Star', 'Country', 'Gross', 'Company', 'Runtime'], axis=1)
new_na_df = new_na_df.astype(np.float64)
orig_na = new_na_df.copy()

#normalization
new_na_df -= new_na_df.min()
new_na_df /= new_na_df.max()

#predict revenue
new_na_pred = reg.predict(new_na_df)

#put data into df 
orig_na = orig_na.to_numpy()
orig_na = orig_na.flatten()

#prediction df
predicted_gross = pd.DataFrame({'Budget': orig_na.round(2) , 'Gross': (new_na_pred * orig_na).round(2)})
na_df_copy['Gross'] = predicted_gross['Gross'].values

#replace gross < 0 
na_df_copy.drop(na_df_copy[na_df_copy['Gross'] < 0].index, inplace=True)

#concat the two dataframes
filled = pd.concat([filled_df, na_df_copy])

#normalized data
normalized_budget = filled['Budget'].astype(float)
normalized_gross = filled['Gross'].astype(float)

#added filled in data to previous plot
normalized_budget = (normalized_budget - normalized_budget.min()) / normalized_budget.max()
normalized_gross = (normalized_gross - normalized_gross.min()) / normalized_gross.max()
plt.scatter(normalized_budget, normalized_gross)
plt.axline((0, 0), slope=1, color = 'r')
plt.xlabel("Budget")
plt.ylabel("Gross")
plt.show()

# %% [markdown]
# At this point, our data has been filled in to the best of our abilities. It is important to note that a majority of movies DO NOT PROFIT, which goes with our model that we found. Looking at the plot above comparing budget to gross, any movie will break even, if it is above the line y = x. This means that a majority of the movies that were in the sample did not profit. With that, we will proceed with completing the initial goal of giving an inherent rating to each director based of their performance.
# 
# pseudo code
# we well be using the following columns:
#   - Director
#   - Rating (score out of 10)
#   - Votes
#   - Total Income (Gross - Budget)
# 
# - separate the mentioned data above, meaning drop the rest of the unused variables
# - Once again do linear regression
#   - directors who show up more than once, simply average their rating
#   - i would like to make it so that it becomes a bell curve
#   - display ratings however
# 

# %%
import matplotlib.pyplot as plt


# %%
#get copy of data from the filled gross in linear regression above
data = filled.copy()

#make dataframe with gross - budget for income
income = (data.loc[:, 'Gross'].astype(float) - data.loc[:, 'Budget'].astype(float))/data.loc[:, 'Budget'].astype(float)
income = pd.DataFrame(income)

#new data
data.drop(columns=['Name', 'Rating', 'Genre', 'Year', 'Released', 'Writer', 'Star', 'Country', 'Budget', 'Gross', 'Company', 'Runtime'], inplace=True)
data.insert(2, 'Income', income)
#dfinal modified data we are working with to form our model
print(data)

# %%
fig = plt.figure()
ax = plt.axes(projection='3d')

#remove outliers
graph_data = data[data['Income'] < 80]

xdata = graph_data.iloc[:, 2].astype(float)
ydata = graph_data.iloc[:, 1].astype(float)
zdata = graph_data.iloc[:, 0].astype(float)

ax.plot3D(xdata, ydata, zdata)
ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis', edgecolor='none')
ax.set_xlabel('Income')
ax.set_ylabel('Votes')
ax.set_zlabel('Score')
ax.view_init(-35, 45)
plt.show()

# %%
Y = data.iloc[:, 0].astype(float)
X = data.iloc[:, 1:4]

#split data in to training and testing data
x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=None)
x_train_names = x_train.iloc[:, 2]
x_test_names = x_test.iloc[:, 2]
x_train = x_train.iloc[:, 0:2].astype(float)
x_test = x_test.iloc[:, 0:2].astype(float)

#normalization
x_train = x_train - x_train.min()
x_train = x_train / x_train.max()
x_test = x_test - x_test.min()
x_test = x_test / x_test.max()
y_train = y_train - y_train.min()
y_train = y_train / y_train.max()
y_test = y_test - y_test.min()
y_test = y_test / y_test.max()

#gradient descent linear regression model
scoreReg = SGDRegressor(max_iter=10000, tol=1e-5).fit(x_train, y_train)
scorePred = scoreReg.predict(x_test)
RMSE = np.sqrt(mean_squared_error(y_test, scorePred))
MAE = mean_absolute_error(y_test, scorePred)

scorePred = pd.DataFrame(scorePred * 10, columns=['Rating'])
scorePred.insert(1, 'Director Name', x_test_names.values)

# %%
from collections import defaultdict

director_rating = defaultdict(tuple)

#tuples read as: (sum, avg, number of movies made)
for i, r in scorePred.iterrows():
  if r['Director Name'] in director_rating:
    sum = director_rating[r['Director Name']][0] + r['Rating']
    count = director_rating[r['Director Name']][2] + 1

    director_rating[r['Director Name']] = (sum, sum / count, count)
  else:
    director_rating[r['Director Name']] = (r['Rating'], r['Rating'], 1)

director_rating = pd.DataFrame([(i, round(v[1],2)) for i , v in director_rating.items()], columns=['Director Name', 'Rating'])
#print director names with their ratings attached
print(director_rating)

# %%
print(RMSE)
print(MAE)

# %%
#counts number of directors of a particular rating, rounded down
rating_data = director_rating['Rating'].apply(np.floor)
count = pd.Series(rating_data).value_counts()
count_df = pd.DataFrame(count)
#prints the count of general ratings of the directors
print(count_df)



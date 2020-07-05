---
title: "PGA Player Winnings by Shot Type"
date: 2020-06-29
tags: [golf analysis, data science, data wrangling, regression]
header: 
    image: "/images/teeoff.jpg"
excerpt: "Golf Analysis, Data Science, Data Wrangling, Regression Analysis" 
mathjax: "true"
---

# Introduction 
Most golfers have heard the saying: 'drive for show, putt for dough'. How valid is this claim? When examining PGA pros and their relevant statistics during a match, we can determine which part of their game generates the largest amount of winnings.

# Data 
The data was obtained from kaggle, and published by Brad Klassen. The data set can be explored by following the link: 
https://www.kaggle.com/bradklassen/pga-tour-20102018-data

# Cleaning
The following code takes the kaggle data and reformats it into wide format in order to run a regression. 

```python
# Import pandas and numpy
import numpy as np 
import pandas as pd 

# Path to kaggle data
train_file = "2019_data.csv"
test_file = "2020_data.csv" 

# Select variables to be used for regression, 
reg_vars = ['Percentage of Yardage covered by Tee Shots - (AVG (%))',
'Going for the Green - Hit Green Pct. - (%)',
'GIR Percentage - 200+ yards - (%)',
'GIR Percentage - 175-200 yards - (%)',
'GIR Percentage - 150-175 yards - (%)',
'GIR Percentage - 125-150 yards - (%)',
'GIR Percentage - < 125 yards - (%)',
'GIR Percentage from Other than Fairway - (%)',
"Putting from - > 25' - (% MADE)",
"Putting from 15-25' - (% MADE)",
"Putting from 5-15' - (% MADE)",
'Official Money - (MONEY)']

# Read in data in order to clean and reformat
df_train = pd.read_csv(train_file,delimiter=',', encoding="utf-8-sig")
df_test = pd.read_csv(test_file,delimiter=',', encoding="utf-8-sig")

# Function to clean and transfor data into wide format 
def pgadatatransform(df):
    # Combine statistics and variables in order to find stats using percentages instead of ranking
    stats = df['statistic'] + ' - (' + df['variable'] + ')'
    del df['variable']
    df['variable'] = stats 
    del df['statistic']
    del df['date']

    # Select variables for regression
    df = df[df['variable'].isin(reg_vars)]
    # Reformat the Money variables into standard numbers and reformat to wide 
    df['value'] = df['value'].str.replace('["$,]', '').astype(float)
    df_wide = df.pivot_table(index=['player_name','tournament'], columns='variable',values='value').reset_index()

    return(df_wide)

# Apply function to both the train and test data frames
df_train = pgadatatransform(df_train)
df_test = pgadatatransform(df_test)  

# Save cleaned data
df_train.to_csv('train.csv')
df_test.to_csv('test.csv')
```  

From here regression analysis can be preformed.  

# Regression  
The following code was used to run a regression: 

```python
# Import the required libraries
from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Select the 2019 data
train = pd.read_csv('train.csv')  

# Function for dropping index column and converting percentage columns to percentages
def regtransform(df):
    df.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
    df.drop(["a"], axis=1, inplace=True)  
    del df['player_name']
    del df['tournament'] 

    df = df.apply(lambda x: x/100 if x.name not in ['Official Money - (MONEY)'] else x)
    df = df.dropna()
    return df

df_train = regtransform(train)

# Seperate dependent variable and independent variables
X = df_train.loc[:, df_train.columns != 'Official Money - (MONEY)'] 
y = df_train['Official Money - (MONEY)']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
lm = linear_model.LinearRegression()
model= lm.fit(X_train, y_train)
predictions = lm.predict(X_test) 

# Print score and coefficients
coefs = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("Score:", model.score(X_test, y_test))
print(coefs)
``` 

The independent variables were all the GIR, putting, and yardage covered by drive percentages. The dependent variablers was official money. Overall the regression did not perform as expected, with extremely low model scores. 

Model scores:

```output
Score: 0.05752240150422671
                                                      Coefficient
GIR Percentage - 125-150 yards - (%)                 13097.686751
GIR Percentage - 150-175 yards - (%)                 58383.831588
GIR Percentage - 175-200 yards - (%)                107007.108932
GIR Percentage - 200+ yards - (%)                    52704.022171
GIR Percentage - < 125 yards - (%)                  178493.525339
GIR Percentage from Other than Fairway - (%)         53885.882518
Going for the Green - Hit Green Pct. - (%)           30034.388464
Percentage of Yardage covered by Tee Shots - (A...  508098.500781
Putting from - > 25' - (% MADE)                     275662.314634
Putting from 15-25' - (% MADE)                      195041.602165
Putting from 5-15' - (% MADE)                       518973.776384
```

# Conclusion 

Even with low model scores we can see that the putting variables made more money than yardage covered. The other important conclusion is that the GIR percentages were the largest factor in money earned. Overall the model was not accurate, but somewhat confirmed the importance of the short game over driving ability.

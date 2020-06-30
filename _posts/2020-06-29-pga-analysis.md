---
title: "PGA Player Winnings by Shot Type"
date: 2020-06-29
tags: [golf analysis, data science, data wrangling, regression]
header: 
    image: "/images/teeoff.jpg"
excerpt: "Golf Analysis, Data Science, Data Wrangling, Regression Analysis" 
mathjax: "true"
---
# PGA Player Winnings by Shot Type

## Introduction 
Most golfers have heard the saying: 'drive for show, putt for dough'. How valid is this claim? When examining PGA pros and their relevant statistics during a match, ew can determine which part of their game generates the largest amount of winnings.


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

# Conclusion 


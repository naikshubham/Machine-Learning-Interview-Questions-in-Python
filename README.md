# Machine-Learning-Interview-Questions-in-Python
Machine Learning Interview Questions in Python

### Handling missing data
- Impact of different techniques
- Finding missing values
- Strategies to handle

#### 1) Omission
- Removal of rows --> `.dropna(axis=0)`
- Removal of columns --> `.dropna(axis=1)`

#### 2) Imputation
- Fill with zero -> `SimpleImputer(strategy='constant', fill_value=0)`
- Impute mean    -> `SimpleImputer(strategy='mean')`
- Impute median  -> `SimpleImputer(strategy='median')`
- Impute mode    -> `SimpleImputer(strategy='most_frequent')`
- Iterative imputation -> `IterativeImputer()` .Imputes by modelling each feature with missing values as a function of other features

- How we handle missing data cann introduce bias, handling it appropriately will reduce the probablity of introducing bias

### Effects of imputation
#### Depends on:
- Missing values
- Original Variance
- Presence of outliers
- Size and direction of skew

- **Omission** : Removing rows or columns might result in removing too much data.
- **Filling with zero** : Tends to bias the results downwards
- **Mean** : Affected more by outliers
- **Median** : Better in case of outliers


Function               | returns
:---------:            | :-------:
df.isna().sum()        | find the number of missing values
df['feature'].mean()   | returns the mean
.fillna(0)             | fills missing values with the arguments passed to it


























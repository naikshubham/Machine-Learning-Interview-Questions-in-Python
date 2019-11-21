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

### Data distributions and transformations
- It can happen that the way we split the dataset before we send it into a machine learning model can cause the train & test sets to have different distributions which introduces bias into the Machine Learning Framework.
- If the distributions have both different mean and variance, this will likely contribute to poor model performance.


#### Train/Test split

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
```

`sns.pairplot()` --> plot matrix of distributions and scatterplots

### Data transformation
- Outliers are one cause of non-normality or non-gaussion behaviour.
- Transformation can reduce the impact of exisiting outliers
- Box-Cox Transformations (power transformation)
`scipy.stats.boxcox(data, lmbda=)`

### Data outliers and scaling
- Outliers : One or more observations that are distant from the rest of the observations.
- Inter-quartile range(IQR) : Defined as the difference at the values at the first and the third quartiles which are at 25% and 75% respectively, with median exactly between at 50%
- Those points above or below (1.5 * IQR) are suspected as outliers

### Outlier functions

Function                            | returns
:---------------------------------: | :----------:
sns.boxplot(x= ,y='Loan status')    | boxplot conditioned on target variable
sns.distplot()                      | histogram and kernel density estimate
np.abs()                            | returns absolute value
stats.zscore()                      | calculated z-score
mstats.winsorize(limits=[0.05,0.05])| given a list of limits replaces outliers(in this e.g with the 5th percentile & 95th percentile data values)
np.where(condition, true, false)    | replaced values

### Variance 
- Deviation from the mean. In machine learning application, a high variance feature will be chosen more often than a low variance feature making it seem more influential when it may not be
- **The solution to this problem is to scale the data when the dataset contains features that vary greatly**

### Standardization vs Normalization
- Sometimes the terms for scaling standardization & normalization are used interchangibly 
- **Standardization** : Standardizing data also known as **Z-score** ,takes each value subtracts by the mean and divides it by the standard deviation, giving it a mean of zero and std-dev =1
```python
z = xi - mean / std-dev
```
- **Normalization** : Also known as **min/max normalizing**, takes each value minus the minimum and divides by the range.This scales the features between 0 & 1.
```python
z = x - min(x) / max(x) - min(x)
```
- Both the approaches are scaling the data, but they just do it differently

### Scaling functions
- sklearn.preprocessing.StandardScaler() --> (mean=0, sd=1)
- sklearn.preprocessing.MinMaxScaler()   --> (0, 1)

#### Preprocessing steps : `Missing --> Transforms --> outliers --> Scaling`

### Handling Outliers
- Visualizing data using boxplot is one way to visualize outliers. Another way for handling outliers is by calculating the Z-score which gives a threshold for outliers approximately +/- 3 standard deviations away from the mean.




















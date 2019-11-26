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

## Regression : feature selection
- Selecting the correct features reduces overfitting by removing unwanted features that contributes noise but not information.
- Improves accuracy since any misleading data is removed
- Since model is less complex it is more interpretable
- Less data means the ML algorithm takes less time to train

### Feature selection methods
- **Filter** : Rank features based on statistical performance between the independent variable and the target variable
- **Wrapper** : Uses an ML model to evaluate performance
- **Embedded** : Iterative model training to extract features
- **Feature importance** : Offered by few of the tree-based ML models in scikit-learn

### Compare and contrast methods
- **Filter methods** are the only techniques that do not use any ML model, but rather **correlation**, this means that the best subset may not always be selected but prevents overfitting. However, the model based methods have the advantage of selecting the best subset, however depending on the parameter this can lead to overfitting.

Method            | Use an ML model | Select best subset | Can overfit |
:---------------: |:---------------:|:------------------:|:-----------:|
Filter            | No              | No                 | No          |
Wrapper           | Yes             | Yes                | Sometimes   |
Embedded          | Yes             | Yes                | Yes         |
Feature Importance| Yes             | Yes                | Yes         |

### Correlation coefficient statistical tests (Filter method) 
- The statistical tests for filter methods depends on the datatype of feature and response.This gives a **numerical relationship** between all the features so that a threshold can be applied as a filter, thus the name **filter method**

Feature/Response | Continuous            | Categorical |
:---------------:|:---------------------:|:-----------:|
Continuous       | Pearson's Correlation | LDA         |
Categorical      | ANOVA                 | Chi-Square  |

### Filter functions

**Function**            | **returns**                  |
:----------------------:|:----------------------------:|
df.corr()               | Pearson's correlation matrix |
sns.heatmap(corr_object)| heatmap plot                 |
abs()                   | absolute value               |

- We also use `abs()` function to return the threshold value since correlation could be negative or positive


### Wrapper methods
1) **Forward selection (LARS - least angle regression)** : Sequentially adds features one at a time based on thier model contribution. **`Starts with no features, adds one at a time`**

2) **Backward elimination** : Starts with all the features, and sequentially drops features based on the least contribution in a given step.

3) **Forward selection/backward elimination combination(bidirectional elimination)** 

4) **Recursive feature elimination (RFECV)**

### Embedded methods
- Includes **Lasso Regression**, **Ridge Regression** and **ElasticNet(hybrid of lasso & ridge)** 
- They perform an iterative process which extracts the features that contributes the most during a given iteration to return best subset dependent on the penalty parameter **alpha**

### Tree-based feature importance methods
- **Random forest** --> `sklearn.ensemble.RandomForestRegressor`
- **Extra Trees** --> `skleanr.ensemble.ExtraTreesRegressor`
- **After model fit** -> `tree_mod.feature_importances_`


Function                             | returns                                     |
:-----------------------------------:|:-------------------------------------------:|
sklearn.svm.SVR                      | support vector regression estimator         |
sklearn.feature_selection.RFECV      | recursive feature elimination with cross-val|
rfe_mod.support_                     | boolean array of selected features          |
rfe_mod.ranking_                     | feature ranking, selected = 1               |
sklearn.linear_model.LinearRegression| linear model estimator                      |
skleanr.linear_model.LarsCV          | least angle regression with cross-val       |
LarsCV.score                         | r-squared score                             |
LarsCV.alpha_                        | estimated regularization parameter          |


### Regression : Regularization algorithms
- **Ridge, Lasso, ElasticNet regressions are forms of regularizations**, simple techniques designed to reduce model complexity and help prevent overfitting. They do so by adding a penalty term to Ordinary Least Squares or OLS formula.
- **Ordinary least squares** : sum(yi - (yi)^)^2 : Minimizes the sum of the square residuals
- **Ridge loss function ** : sum(yi - (yi)^)^2 + lambda sum(beta^2) (Ridge penalty term)
With Ridge the penalty term is added by multiplying penalty parameter lambda times the squared coefficent values(Beta). This shrinks the value of the coefficents towards zero, but not zero which is called L2 regularization or L2 norm

- ** Lasso loss function** : sum(yi - (yi)^)^2 + lambda sum(abs(beta)) (Lasso penalty term) Also called L1 regularization or L1 norm is similar to ridge except that it takes the absolute value of the coefficents instead of squares, this results in shrinking less important feature coefficents to zero and results in a type of feature selection 

### Ridge Vs Lasso
|Regularization     |            L1(Lasso)                  |       L2(Ridge)               |
|:-----------------:|:-------------------------------------:|:-----------------------------:|
|penalizes          |sum of absolute values of coefficients |sum of squares of coefficients |
|solutions          |sparse                                 | non-sparse                    |
|number of solutions| multiple                              | one                           | 
|feature selection  | yes                                   | no                            |
|robust to outliers?| yes                                   | no                            |
|complex patterns?  | no                                    | yes                           |

### ElasticNet 
- Hybrid of Ridge and Lasso, using an L1 ratio. ElasticNet combines the two penalization methods with penalty as L2 when the L1 ratio is zero and L1 when its 1. This allows flexible regularization anywhere between lasso and ridge. lambda is a shared penalization parameter while alpha sets the ratio between L1 & L2 regularization parameter. Alpha is set automatically.

### Regularization with Boston housing data
- Predicts house prices given several features

|        Features                 |CHAS | NOX |RM  |
:--------------------------------:|:---:|:---:|:--:|
|Coefficient estimates            |2.7  |-17.8|3.8 |
|Regularized coefficient estimates|0    | 0   |0.95|
- Removing this unimportant features results in less noise and higher accuracy

### Regularization functions

```python
# Lasso estimator
sklearn.linear_model.Lasso

# Lasso estimator with cross-validation
sklearn.linear_model.LassoCV

# Ridge estimator
sklearn.linear_model.Ridge

# Ridge estimator with cross-validation
sklearn.linear_model.RidgeCV

#ElasticNet estimator
sklearn.linear_model.ElasticNet

#ElasticNet estimator with cross-validation 
sklearn.linear_model.ElasticNetCV

# Train-test split
sklearn.model_selection.train_test_split

# Mean squared error
sklearn.metrics.mean_squared_error(y_test, predict(X_test))

# Best regularization parameter
mod_cv.alpha_

# array of log values
alphas=np.logspace(-6, 6, 13)
```




























































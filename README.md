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

SrNo|Steps
:--:|:---------------------------------------:
i)  | Start with an empty feature set
ii) |Try each feature one by one
iii)| Estimate the accuracy i.e estimate the classification or regression error on adding each feature
iv) |Select feature that gives maximum improvement

2) **Backward elimination** : Starts with all the features, and sequentially drops features based on the least contribution in a given step.

sr no. | steps
:-----:|:--------------------:
i)     | Start with the whole feature set
ii)    |Try removing features
iii)   |Find that feature whose removal gives rise to maximum improvement in the performance, and drop that feature


3) **Forward selection/backward elimination combination(bidirectional elimination)** 

4) **Recursive feature elimination (RFECV)** : A variant of multivariate feature selection

Sr no | steps
:----:|:---------------:
i)    |Compute weights on all the features
ii)   |Remove features with smallest weights
iii)  |Recompute weights on reduced data
iv)   |If stopping criteria not met repeat step2

5) **Univariate analysis** : Looks at each feature independently. We can build a model using individual feature and based on the model evaluation metric such as accuracy we can determine whether the feature is useful in predicting targets or not.

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
With Ridge the penalty term is added by multiplying penalty parameter lambda times the squared coefficient values(Beta). This shrinks the value of the coefficients towards zero, but not zero which is called L2 regularization or L2 norm

- ** Lasso loss function** : sum(yi - (yi)^)^2 + lambda sum(abs(beta)) (Lasso penalty term) Also called L1 regularization or L1 norm is similar to ridge except that it takes the absolute value of the coefficients instead of squares, this results in shrinking less important feature coefficients to zero and results in a type of feature selection 

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

### Classification : Feature Engineering
- Extracts additional information from the data
- Creates additional relevant features
- One of the most effective ways to improve predictive models

### Benefits of feature engineering
- Increased predictive power of the learning algorithm
- Makes your machine learning models perform even better!

### Types of feature enginnering
- **Indicator variables**
- **Interaction features**
- **Feature representation**

### `Indicator varaibles`
- **Threshold indicator** : Example of a Threshold indicator is when we use a **feature such as age** to distinguish whether a value is above or below a given threshold like high school Vs college
- **Multiple features** : can be used as a flag to indicate properties if we have the domain knowledge, that the combination is considered premium
- **Special events** : such as Black Friday or Christmas
- **Groups of classes** : can be used to create a paid flag for website traffic sources such as Google adwords or Facebook ads

### `Interaction features`
- **Are created by using two or more features** and then taking thier sum, difference, product, quotient, other mathematical combos. Since combined features may predict better then separately

### `Feature Representation`
- Take Datetime stamps and extract the day of week or hour of day
- Grouping categorical levels into small number of observations as a single level called 'Other'
- Transformation of categorical variables to dummy varaibles commonly called **one hot encoding** 

### Different categorical levels
- We need to take care of the classes that exists in training data but not in test data.
- Training data: model trained with [red, blue, green]
- Test data : model test with [red, green, yellow], additional color not seen in training.
- We can tackle this using **robust one-hot encoding**

### Engineer feature using Debt to income ratio
- Using original features Monthly Debt and Annual Income/12

### Feature engineering fucntions
| Function                                              | returns                  |
|:-----------------------------------------------------:|:------------------------:|
| sklearn.linear_model.LogisticRegression               | logistic regression      |
| sklearn.model_selection.train_test_split              |train/test split function |
| sns.countplot(x='Loan status', data=data)             |bar plot                  |
| df.drop(['Feature 1', 'Feature 2'], axis=1)           |drops list of features    |
| df['Loan Status'].replace({'Paid':0,'Not Paid':1})    |Loan status               |
| pd.get_dummies()                                      |k-1 binary features       |
| sklearn.metrics.accuracy_score(y_test,predict(X_test))|model accuracy            |

   
### Ensemble methods
- Ensemble learning techniques : **Bootstrap Aggregation (a.k.a Bagging), Boosting, Model Stacking**
- A **linear model**, no matter how it is fit to the data it is unable to capture the true curved relationship
- **Bias** is the inability for the Machine Learning Method to capture the true realtionship
- Making the assumption that data has a linear relationship when it's actually more complex results in **high bias** , **underfitting the model** and **poor model generalization**
- **Bias decreases** when the model complexity increases, since the model tends towards accurate representation of the complex structure in the data. 
- **Complex models fits the train points, but fails to fit the test points**
- **Variance** : Algorithms in High complexity models tends to model random noise in training data creating a large difference in fits between training and test datasets. As more complex structures are identified, sensitivity towards small changes in data also increases leading to **high variance, overfitting and poor model generalization**

### Bias-Variance Trade-Off
- In Machine Learning the best algorithm has a **low bias** which can accurately model the true relationship, but also has **low variance** meaning it can provide consistent predictions with different datasets. This is the sweet spot we seek to find in ML achieving the lowest bias and lowest variance.

### Bagging (Bootstrap aggregation)
- Uses Bootstrap sampling. Bootstrapping is an sampling techinque where the subset of the data is selected with replacement, meaning the same row of data maybe chosen more than once in a given subset.
- **Model** is built for each bootstrap sample and output predictions are averaged which reduces variance and produces an more accurate model.

### Boosting
- Boosting also builds multiple sequential models but does so in a sequential order. Learning to reduce predictive errors from previous models by modifying the original dataset with weights for incorrectly predicted instances.
- Results in a model with decreased bias.

### Model stacking
- Takes the predictions from individual models and combines them to create an higher accuracy models.
- Model stacking uses predictions of base classifiers as inputs for training to a second level model. 

### Vecstack Package
- Vecstack package contains an convenient function called `stacking`, which takes the list of **instantiated models, X_train, Y_train and X_test**, it then outputs a set of objects that can conveniently be used in a second level modelling

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from vecstack import stacking

# create list : stacked_models
stacked_models = [BaggingClassifier(n_estimators=25, random_state=123), AdaBoostClassifier(n_estimators=25, random_state=123)]

# stack the models : stack_train, stack_test
stack_train, stack_test = stacking(stacked_models, X_train, y_train, X_test, regression=False, mode = 'oof_pred_bag', needs_proba=False, metric=accuracy_score, n_folds=4, stratified=True, shuffle=True, random_state=8, verbose=2)

# initialize and fit 2nd level model
final_model = XGBClassifier(random_state=123, n_jobs=-1, learning_rate=0.1, n_estimators=10, max_depth=3)
final_model_fit = final_model.fit(stack_train, y_train)

# predict : stacked_pred
stacked_pred = final_model.predict(stack_test)

# final prediction score
print("Final prediction score: [%.8f]' % accuracy_score(y_test, stacked_pred))
```

### Ensemble functions
|Algorithm             | Function                             |
|:--------------------:|:------------------------------------:|
|Bootstrap aggregation | sklearn.ensemble.BaggingClassifier() |
|Boosting              | sklearn.ensemble.AdaBoostClassifier()|
|XGBoost               | xgboost.XGBClassifier()              | 

### Bagging Vs boosting
|Technique        | Bias    | Variance |
:----------------:|:-------:|:--------:|
| Bagging         |Increase | Decrease |
| Boosting        |Decrease | Increase |


### Unsupervised learning methods
Unsupervised learning is used to find the patterns in the data using Principal Component Analysis(PCA) and Singular Value decomposition(SVD).**Dimensionality Reduction != Feature selection** . Dimensionalty reduction techniques creates new combination of features. Feature selection includes or excludes features based on its relationship to the target variable, but there is no feature transformation happening as there is with dimensionality reduction.

### Curse of dimensionality
We start with as many features as possible which is the best practice, however there is a phenomenon which happens where the **model performance decreases as the number of features increase**. This is so called curse of dimensionality. In a high dimensionality context the feature space becomes more sparse leading to **overfitting**.

### Dimension reduction function

| Function/method                      |   returns                               |
:-------------------------------------:|:---------------------------------------:|
|sklearn.decomposition.PCA             |principal component analysis             |
|sklearn.decomposition.TruncatedSVD    |singular value decomposition             |
|PCA/SVD.fit_transform(X)              |fits and transforms data                 |
|PCA/SVD.explained_variance_ratio_     |variance explained by Principal component|

### Dimensionality Reduction : Visualization techniques
- **Speeds up ML training** : Less dimension means algorithms can simply run faster
- **Visualization** : Helps us to visualize the data, since visualizing more than 3 dimensions are troublesome
- **Improves accuracy** : It improves the accuracy of our trained models bcz it removes unimportant and colinear information resulting in less noise and redundancy which results in more acurately trained models.

### Visualizing with PCA (Mathematical technique)
- **First principal component** is a linear combination of original features **which captures the maximum variance** in the dataset and **determines the direction of highest variability** 
- **Second Principal component** is also a linear combination of original features capturing remaining variance in the dataset and is **uncorrelated with the first PC**
- Using the **`explained_variance_ratio_`** method we printed the variance. This variance ratio when plotted creates what is called as a **Scree plot**. This helps to visually determine how many PC it takes to describe the maximal amount of variance or information contained in the dataset. It can be used to obtain the **optimal number of PC** to take forward for modelling.

### Visualizing with t-SNE (Probabilistic technique)
- It takes pairs of data points in high dimensional space and computes a probability that they are related and chooses a low dimensional embedding to produce a similar distribution. These embeddings then can be visualized.

### Clustering algorithms
Common practical applications of clustering are Customer Segmentation, Document Classification, Detection of anamolies like (Insurance/transcation fraud detection), Image segmentation, Anomaly detection etc.

#### K-means
- **1)** Choosing the initial centroids or choosing the location of the cluster.
- **2)** Then each observation is assigned to its nearest centroid
- **3)** Followed by taking the mean of all the obsevations assigned to a given centroid to create new centroids
- **4)** Steps 2 and 3 are iterated until the centroids significantly move.

#### Hierarchical agglomerative clustering
- Involves successivley merging or splitting observations. The hierarchy is represented as a tree known as a **dendrogram**.
- Agglomerative clustering uses an bottom up approach where each observation starts in its own cluster becoming merged into groups of clusters based on a given linkage criteria. Using **dendogram** to select the number of clusters depends on both the linkage criteria and the distance threshold.

### Selecting a clustering algorithm
- There is no best way to select clustering algorithm, however one way is to assess cluster stability which can be done by **comparing algorithms that share some similarity**. E.g K-means and HC both use Euclidian distance and are therefore comparable.
- Intra-cluster distances:mean of the distance between points of a cluster and cluster centroid & Inter cluster distances can be computed as a mean of the distances between cluster centroids. **In any well formed cluster the intra cluster distances should be less than the inter cluster distances** 
- When comparing different clustering algorithms, you must make sure they are comparable by assessing the same distance metric between them, not different distance metrics.

#### Clustering functions 

|Functions                                 |returns                                                                 |
:-----------------------------------------:|:----------------------------------------------------------------------:|
|sklearn.cluster.Kmeans                    |K-Means clustering algo                                                 |
|sklearn.cluster.AgglomerativeClustering   |Agglomerative clustering algo                                           |
|kmeans.inertia_                           |SS distances of observations to closest cluster center                  |
|scipy.cluster.hierarchy as sch            |Hierarchical clustering for dendograms                                  |
|sch.dendrogram()                           |Dendogram function                                                      |
|KMeans.inertia_                           |**Sum of squared distances of samples to their closest cluster center** |

### Choosing the optimal number of clusters particularly for K-Means
- Two most used methods are **Silhouette method** & **Elbow method**

#### Silhouette method
- **Silhouette method** : uses the **silhouette coefficient** which is composed of two scores, **1)** The mean distance between the observation and all of the other observations **in the same cluster** and **2)** the mean distance betwn the observation and all other observations **in the next nearest cluster**
- The silhouette coefficient values are between -1 and 1, with **1 denoting the observation is very near others in the same cluster and very far from others in the other clusters** 
- **-1** is the worst score and the observation is not near others in the same cluster and close to others in the other cluster and may have actually been assigned to the wrong cluster.
- Score of **0** indicates that there is a overlap among the clusters. The observation is on or close to boundary between two clusters.
- There is a convinient function from sklearn.metrics called silhouette_score which called on a data matrix and labels of a trained k-means model returns the mean silhouette coefficient of all observations as one simple to interpret score 

#### Elbow method
- Is a visualizing technique and the resulting plot looks like an arm, then the elbow on the arm looks like the optimal k. It uses the sum of the squared distances from each observation to its nearest cluster center or centroid similar to **inertia** attribute from a trained k-means model.The sum of squares decreases as the values for k increases.

### Optimal K selection functions
| Function                        | returns                                                          |
:--------------------------------:|:-----------------------------------------------------------------:
|sklearn.cluster.KMeans           |K-Means clustering algorithm                                      |
|sklearn.metrics.silhouette_score |score between -1 and 1 as measure of cluster stability            |
|kmeans.inertia_                  |sum of squared distances of observations to closet cluster center |


### Model generalization : bootstraping and cross-validation
- **Model generalization** : A ML model's ability to perform well on unseen data test dataset and future data. Similar evaluation metric between train and test set are the symbol of model generalizabilty.

#### Decision Trees
- **Bootstrapping** is one of the methods that helps with model generalization. Decision trees is a supervised learning algorithm used to build predictive machine learning models for both categorical and continous target variables.

<p align="center">
  <img src="/data/DT.JPG" width="350" title="Decision Tree">
</p>

- The top of the DT is called the root node, where we see a split was made with X3<=0.8. The criteria used to make that split is **Gini index**. It is the measure of impurity. **Samples** corresponds to the number of observations in the dataset.The **value** is the number of observations in each class.
- The observations that are evaluated True are split to the left, the remainder going to the right. And it continues until the decision is made making splits and directing observations.
- If this was **regression tree** instead the split criteria would you use the lowest mean square error to make splits rather than gini.
- **Advantages** : Easy to understand and visualize.
- **Disadvantages** : Easily overfit, they are considered greedy(they may not return globally optimal trees) and they are biased in cases of class imblance.

#### Random Forest
- Bootstraped version of many Decision Trees. Bootstrapping is the sampling technique where subset of the data is selected with replacement, averaging the output predictions to reduce variance resulting in more accurate model.

### K-fold cross validation
- K-fold CV is another technique we can use to help model generalize, since its prevents model overfitting. The way it works is training data is split into k-folds . 1-fold is held out and used as a test set, while the remaining folds are used for model training, this continues in an iterative manner until all of the folds have been used as the testing set.

#### Functions

```python
#decision tree
sklearn.tree.DecisionTreeClassifier

#random forest
sklearn.ensemble.RandomForestClassifier

#cross-validated grid search
sklearn.model_selection.GridSearchCV

#model accuracy
sklearn.metrics.accuracy_score

#train test splits
sklearn.model_selection.train_test_split

#parameter that gave best results
cross-val_model.best_params_

# Mean cross-validated score of estimator with best params
cross-val_model.best_score_
```

### GridSearchCV vs RandomSearchCV
- GridSearchCV function test parameters for a given spacing inorder to give paramter estimates. Since this doesn't search entire space its good to be aware of the RandomSearchCV function as its more likely to come up with the optimal paramter estimation. A randomsearch space tends to have a longer runtime.

### Model evaluation : imbalanced classification models
- **Class imbalance** : Imbalance class problem implies that the ML model has an categorical target variable.Most ML algo works best when there are approx equal number of observations in the classes
- When there is a large difference in the number of observations in each class it can cause misleading results
- Confusion matrix shows the number of correctly and incorrectly classified observations in each class.
- **When evaluating the model with imbalance classes accuracy in not the best metrics to use**. A closer look at **confusion matrix** can be insightful and used to calculate better metric in the case of imbalanced classes.

- **Precision** : TP/(FP+TP) measures how often the model is correct, when it predicts the positive class. Low precision indicates a high number of false positives.
- A low precision score indicates that there are too many false positives, bringing the calculation down. Seeking to reduce the number of false positives to increase the precision can be accomplished with trying different classification algorithms and/or resampling techniques.
- **Recall/Sensitivity** : is a measure of how often a positive is predicted when an observation is positive (TP/(TP+FN)). Low recall indicates a high number of false negatives.
- **F1 score** : weighted average of precision and recall also called the harmonic mean of precision and recall. 2*(precision * recall)/(precision * recall)

- Resampling is a technique which tries to create balance between classes. Either oversample minority class or undersample majority class. Always split into train and test before trying oversampling techniques.
- Oversampling before spliting the data can allow the exact same observations to be present into train and test set and lead to overfitting and poor generalization.

#### Functions

```python
sklearn.linear_model.LogisticRegression            #returns Logistic Regression 
sklearn.metrics.confusion_matrix(y_test, y_pred)   #returns confusion matrix
sklearn.metrics.precision_score(y_test, y_pred)    #returns precision
sklearn.metrics.recall_score(y_test, y_pred)       #returns recall
sklearn.metrics.f1_score(y_test, y_pred)           #returns f1 score 
sklearn.utils.resample(deny,n_samples=len(approve))#returns resamples
```

### Model Selection : Regression models
- **Multicollinearity** : Multicollinearity is when independent variables are highly correlated. 
- One of the outputs of the regression models are the estimated regression coefficients. These coefficients are interpreted as the amount of change of the Dependent variables that can be explained by Independent variable while holding all other variables constant in a multiple regression framework.
- But when independent variables are correlated all of the sudden interpreting the amount of explained variance gets less clear, threatning the results of the linear regression analysis.

### Effects of multicollinearity
- Multicollinearity can affect ML models by reducing coefficients and p-values causing variance to be unpredictable.
- Overfitting
- Increased standard error which lowers statistical significance further leading to failing to reject which is a type two error for hypothesis testing.
- True relationship with target variable becomes unclear

### Techniques to address multicollinearity
- We need to identify if multicollinearity exists in our data and do something about it.
- First thing we need to do is create a correlation matrix. Plot a heatmap of correlations to get better visual understanding
- Calculate the variance inflation factor (VIF)
- We can introduce penalizations (Ridge ,Lasso)
- We can do PCA,since it can remove multicollinearity

#### Variance inflation factor
- VIF is another way to determine whether or not features are collinear(correlated) with each other. Higher to correlation, higher the VIF value.Values between 1 to 5 can be safely ignored.

| VIF value | Multicollinearity |
|:---------:|:-----------------:|
|<=1        | no                |
|>1         |yes, but can ignore|
|>5         |yes,need to address|

### Functions

| Functions                             | returns                    |
|:-------------------------------------:|:--------------------------:|
|sklearn.linear_model.LinearRegression  | Linear Regression          |
|data.corr()                            |correlation matrix          |
|sns.heatmap(corr)                      |heatmap of correlations     |
|mod.coef_                              |estimated model coefficients|
|mean_squared_error(y_test,y_pred)      |MSE                         |
|r2_score(y_test, y_pred)               |R-squared score             |
|df.columns                             |column names                |

#### Addressing multicollinearity

After careful exploratory data analysis, you realize that your baseline regression model suffers from multicollinearity. How would you check if that is true or not? Without losing any information, can you build a better baseline model?
- Create a correlation matrix and/or heatmap, then engineer features to combine multicollinear independent variables, making sure to remove the individual features used to create any new features.
- Create a correlation matrix and/or heatmap, then perform Ridge regression to penalize multicollinear independent variables and perform feature selection for modeling.
- Create a correlation matrix and/or heatmap, then perform PCA to combine multicollinear independent variables as new principal components.

### Model Selection : Ensemble models
- How to choose among different ensemble models
- Random forest, Gradient Boosting


#### RF  Vs GB (defaults)

|parameter    | Random Forest   | GradientBoosting |
|:-----------:|:---------------:|:----------------:|
|n_estimators |10               |100               |
|criterion    |gini(or entropy) |friedman_mse      |
|max_depth    |None             | 3                |
|learning_rate|N/A              |0.1               |

- **n_estimators** : For RF indicates the number of trees the algo should build. For GB this is the number of boosting staging to perform
- **criterion** : tells the algo on what basis to split on
- **max_depth** : For RF defaults to None which allows the nodes to expand until all leaves are pure. For GB this is the max depth of individual regression estimators, max_depth limits the nodes in the tree and default is 3
- **learning_rate** : In GB LR shrinks the contribution of each tree by this value. There is a tradeoff between learning_rate and n_estimators parameters. In RF its not applicable.

#### While GB can use any algorithm, RF uses decision trees!

#### Functions 

|Functions                                  | Returns
|:-----------------------------------------:|:----------------------:|
|sklearn.ensemble.RandomForestClassifier    |Random Forest           |
|sklearn.ensemble.GradientBoostingClassifier|Gradient Boosted Model  |














































































































































































































































































































































import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# 1: Introducing Linear Regression
# Load Housing Dataset into df
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

# df = pd.read_csv(('./housing.data.txt'), sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()



# Visualizing the important characteristics of a dataset
# Exploratory Data Analysis; only plotted 5 columns: LSTAT, INDUS, NOX, RM, and MEDV.
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()

# Looking at relationships using Correlation Matrix
# Use Seaborn's heatmap function to plot the correlation matrix array as a heat map
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.show()



# 2: Implementing an ordinary least squares linear regression model
# Solving regression for regression parameters with gradient descent
class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

# LinearRegressionGD: use RM exploratory variable to train model that can predict MEDV (prices)
# standardize variables for better convergence of the GD algorithm
X = df[['RM']].values
y = df['MEDV'].values

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# Check algorithm converged to a cost minimum (here, a global cost minimum)
sns.reset_orig()    # resets matplotlib style
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()  # Will see GD convergence around 5th epoch

# Define helper fxn to visualize how the linear regression line fits in the training data
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average Number of Rooms [RM] (Standardized)')
plt.ylabel('Price in $1000s [MEDV] (Standardized)')
plt.show()

# Scale predicted price outcome using the inverse_transform method of the StandardScaler
num_rooms_std = sc_x.transform([[5.0]])
price_std = lr.predict(num_rooms_std)
print("Price in $1000s: %.3f" %
      sc_y.inverse_transform(price_std))
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


# 3: Estimating coefficient of a regression model via scikit-learn
# More efficient implementations; Scikit LIBLINEAR
slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

# Comparison to GD Implementation
lin_regplot(X, y, slr)
plt.xlabel('Average Number of Rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()  # Should look almost identical to GD implementation, scaled differently

# Closed-Form solution for solving OLS
# adding a column vector of "ones"
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))
print('Slope: %.3f' % w[1])
print('Intercept: %.3f' % w[0])



# 4: Fitting a robust regression model using RANSAC
# Look at RANSAC Algorithm; fits regression model to a subset of data (inliers)
ransac = RANSACRegressor(LinearRegression(),
                         max_trials=100,
                         min_samples=50,
                         loss='absolute_loss',
                         residual_threshold=5.0,
                         random_state=0)
ransac.fit(X, y)

# Obtain inliers and outliers from fitted RANSAC-linear regression model and plot them with linear fit
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white',
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white',
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average Number of Rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.show()  # Linear Regression model will be fitted to set of inliers

# linear regression line is slightly different from the fit that we obtained in the previous section without using RANSAC
print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)




# 5: Evaluating the performance of linear regression models
# Use all variables in the dataset and train multiple regression model
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training Data')
plt.scatter(y_test_pred, y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()  # Should see residual plot with line passing through on x-axis origin

# MSE; compute MSE of training and test predictions
print('MSE Train: %.3f, Test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))
# R^2 on test data; MSE tends to over-fit test data; R^2 is just rescaled version of MSE
print('R^2 Train: %.3f, Test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))

# Using regularized methods for regression
# Ridge, Lasso, and Elastic model initialization
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)


# 6: Turning a linear regression model into a curve – polynomial regression
# Adding Polynomial Terms using Scikit-Learn; add quadratic term d=2 to simple regression problem w/ 1 explanatory variable

# Step 1: Add Second Degree polynomial
X = np.array([258.0, 270.0, 294.0, 320.0, 342.0,
              368.0, 396.0, 446.0, 480.0, 586.0])\
    [:, np.newaxis]

y = np.array([236.4, 234.4, 252.8, 298.6, 314.2,
              342.2, 360.8, 368.0, 391.2, 390.8])
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

# Step 2: Fit simple linear regression model for comparison
lr.fit(X, y)
X_fit = np.arange(250, 600, 10) [:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# Step 3: Fit multiple regression model on the transformed features for polynomial regression
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# Step 4: Plot the results
plt.scatter(X, y, label='Training Points')
plt.plot(X_fit, y_lin_fit,
         label='Linear Fit', linestyle='--')
plt.plot(X_fit, y_quad_fit,
         label='Quadratic Fit')
plt.legend(loc='upper left')
plt.show()  # polynomial fit captures the relationship between the response and explanatory variable much better than the linear fit

y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
print('Training MSE Linear %.3f,\nQuadratic: %.3f' % (
    mean_squared_error(y, y_lin_pred),
    mean_squared_error(y, y_quad_pred)))    # MSE decreased from 570 (linear fit) to 61 (quadratic fit);
                                            # coefficient of determination reflects a closer fit of the
                                            # quadratic model (R2 = 0.982) as opposed to the linear fit (R2 = 0.832)



# 7: Modeling nonlinear relationships in the Housing dataset
# Apply non-linear models to Housing data
X = df[['LSTAT']].values
y = df['MEDV'].values

regr = LinearRegression()

# Create Quadratic Features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# Fit Features
X_fit = np.arange(X.min(), X.max(), 1) [:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# Plot Results
plt.scatter(X, y, label='Training Points', color='lightgray')
plt.plot(X_fit, y_lin_fit,
         label='Linear (d = 1), $R^2 = %.2f$' % linear_r2,
         color='blue',
         lw=2,
         linestyle=':')
plt.plot(X_fit, y_quad_fit,
         label='Quadratic (d = 2), $R^2 = %.2f$' % quadratic_r2,
         color='red',
         lw=2,
         linestyle='-')
plt.plot(X_fit, y_cubic_fit,
         label='Cubic (d = 3), $R^2 = %.2f$' % cubic_r2,
         color='green',
         lw=2,
         linestyle='--')
plt.xlabel('% Lower Status of the Population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')
plt.show()  # cubic fit captures the relationship between house prices and LSTAT better than the linear and quadratic fit
            # Adding more polynomial features increases complexity of model
            # Increases chance of overfitting

# Test Hypothesis of relationship being looking like f(x) = 2^-x; assuming log-transformation: log(f(x)) = -x
# Transform features
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# Fit Features
X_fit = np.arange(X_log.min() - 1,
                  X_log.max() + 1, 1) [:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

# Plot Results
plt.scatter(X_log, y_sqrt,
            label='Training Points',
            color='lightgray')
plt.plot(X_fit, y_lin_fit,
         label='Linear (d = 1), $R^2 = %.2f$' % linear_r2,
         color='blue',
         lw=2)
plt.xlabel('log(% Lower Status of the Population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc='lower left')
plt.show()  # able to capture the relationship between the two variables with a linear regression line that seems
            # to fit the data better ( R2 = 0.69 ) than any of the polynomial feature transformations




# 8: Dealing with nonlinear relationships using random forests
# Decision Tree Regression (pg 340)
# Use DecisionTreeRegressor to model non-linear relationship between MEDV adn LSTAT
X = df[['LSTAT']].values
y = df['MEDV'].values
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% Lower Status of the Population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()  # captures the general trend in the data;
            # does not capture the continuity and differentiability of the desired prediction



# 9: Random Forest Regression
# Use all features of Housing dataset; train on 60% of model; test on 40%
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.4,
                                                    random_state=1)
forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print('MSE Train: %.3f, Test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))       # Over fits; still explains relationship between variables well

# look at residuals
plt.scatter(y_train_pred,
            y_train_pred - y_train,
            c='steelblue',
            edgecolor='white',
            marker='o',
            s=35,
            alpha=0.9,
            label='Training Data')
plt.scatter(y_test_pred,
            y_test_pred - y_test,
            c='limegreen',
            edgecolor='white',
            marker='s',
            s=35,
            alpha=0.9,
            label='Test Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10, 50])
plt.show()  # Model fits training data better than test data; distribution of residuals does not
            # seem to be random around zero point center
            # Model is not able to capture all the exploratory information
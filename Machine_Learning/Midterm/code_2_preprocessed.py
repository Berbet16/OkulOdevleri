import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ------------------------STEP 1-------------------------------------------
# Read Data
df = pd.read_excel('C:/Users/BS/Desktop/MidtermReport_171805019_BetulBernaSoylu/RealEstateRegression.xlsx', usecols=('C:H'))
# -----------------------------------------------------------------------------


# ------------------------STEP 2-------------------------------------------
# preproccesed halde.
preprocessing_data = pd.DataFrame(StandardScaler().fit(df).transform(df), columns=('X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude','Y house price of unit area'))
# -----------------------------------------------------------------------------


# ------------------------STEP 3-------------------------------------------
#giriş değerlerinin dağılımlarının çizdirilmesi
preprocessing_hist = preprocessing_data[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores','X5 latitude', 'X6 longitude' ]]
preprocessing_hist.hist(bins=5,figsize=(16,9),grid=False,);
plt.xlabel('X2 house age')
plt.xlabel('X3 distance to the nearest MRT station')
plt.xlabel('X4 number of convenience stores')
plt.xlabel('X5 latitude')
plt.xlabel('X6 longitude')
plt.show()
# -----------------------------------------------------------------------------

# ------------------------STEP 4-------------------------------------------
X=preprocessing_data.iloc[:,:-1].values
Y=preprocessing_data['Y house price of unit area'].values

#separates data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

#Linear regression algorithm
lr = LinearRegression()

#Ridge regression algorithm
ridge_model = Ridge(alpha=0.01)

#Lasso Regression Algorithm
lasso_model = Lasso(alpha=0.01)

#K-NN Regression Algorithm
knn_model = KNeighborsRegressor(n_neighbors=3)

#ElasticNet Regression Algorithm
elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.5)

#Decision Tree Algorithm
decision_model = DecisionTreeRegressor()

#Random Forest Algorithm
random_model = RandomForestRegressor()

#Support Vector Regressor Algorithm (SVR)
support_model = SVR()
# -----------------------------------------------------------------------------


# ------------------------STEP 5-------------------------------------------
#calculate r2 and mse value
def scoreResults(model, X_train, X_test, Y_train, Y_test):

    Y_train_predict = model.predict(X_train)
    Y_test_predict = model.predict(X_test)

    r2_train = r2_score(Y_train, Y_train_predict)
    r2_test = r2_score(Y_test, Y_test_predict)

    mse_train = mean_squared_error(Y_train, Y_train_predict)
    mse_test = mean_squared_error(Y_test, Y_test_predict)

    return [r2_train, r2_test, mse_train, mse_test, Y_train_predict, Y_test_predict]


# Makes 10-fold process
def regressionAlgrithm(reg_model, X_train, X_test, Y_train, Y_test, reg_name):
    model = reg_model
    k = 10
    iter = 1
    cv = KFold(n_splits=k, random_state = 0, shuffle=True)
    print(reg_name, "Scores")
    for train_index, test_index in cv.split(X):
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        model.fit(X_train, Y_train)
        
        result = scoreResults(model = model
                          ,X_train = X_train
                          ,X_test = X_test
                          ,Y_train = Y_train
                          ,Y_test = Y_test)
        
        print(f"{iter}. veri kesiti")
        print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
        print(f"Test R2 Score: {result[1]:.4f} MSE: {result[3]:.4f}\n")
        iter += 1        
    print("------------------------------------------------------------------------")
    
    
   
regressionAlgrithm(lr, X_train, X_test, Y_train, Y_test, 'Linear Regression Algorithm')
regressionAlgrithm(ridge_model, X_train, X_test, Y_train, Y_test, 'Ridge Regression Algorithm')
regressionAlgrithm(lasso_model, X_train, X_test, Y_train, Y_test, 'Lasso Regression Algorithm')
regressionAlgrithm(knn_model, X_train, X_test, Y_train, Y_test, 'KNN Regression Algorithm')
regressionAlgrithm(elastic_model, X_train, X_test, Y_train, Y_test, 'Elastic Net Regression Algorithm')
regressionAlgrithm(decision_model, X_train, X_test, Y_train, Y_test, 'Decision Tree Regression Algorithm')
regressionAlgrithm(random_model, X_train, X_test, Y_train, Y_test, 'Random Forest Regression Algorithm')
regressionAlgrithm(support_model, X_train, X_test, Y_train, Y_test, 'Super Vector Regression Algorithm')

# draw histograms
models = []
models.append(('LR', lr))
models.append(('RIDGE', ridge_model))
models.append(('LASSO', lasso_model))
models.append(('KNN', knn_model))
models.append(('ELASTIC', elastic_model))
models.append(('DECISIONTREE', decision_model))
models.append(('RANDOMFOREST', random_model))
models.append(('SVR',support_model))

valuesMSE = []
valuesR2 =[]
names=[]

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    
    resultsMSE = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
    resultsR2 = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='r2')
    
    valuesMSE.append(resultsMSE)
    valuesR2.append(resultsR2)
    names.append(name)
    mse = "MSE - %s: %f" % (name, abs(resultsMSE.mean()))
    r2  = "R2 - %s: %f" % (name, abs(resultsR2.mean()))
    print(mse)
    print(r2)

print("\n-------------------------------------------------------------------")
puan=[]
for i in range(len(names)):
    puan.append(abs(valuesMSE[i].mean()))
print("\nBest learning algorithm (Accuracy):")
print(names[puan.index(min(puan))], "->", min(puan))

#Plotting histogram of MSE and R2
r2_hist = plt.hist(valuesR2)
plt.title('Histogram of R2 results')
plt.show()

mse_hist = plt.hist(valuesMSE)
plt.title('Histogram of MSE results')        
plt.show()
# -----------------------------------------------------------------------------


# ------------------------STEP 6-------------------------------------------
#residu and curve plot
X_raw=preprocessing_data.iloc[:,:-1].values
Y_raw=preprocessing_data['Y house price of unit area'].values

#separates data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X_raw, Y_raw, test_size = 0.2, random_state=0)

fig = plt.figure()
fig.suptitle("Algorithm Comparision")

nax=len(models)
i=1
for name, model in models:
    model.fit(X_train, Y_train)
    Y_test_pred = model.predict(X_test)
    
    curveaxis=np.zeros((100, X_test.shape[1]))
    for cx in range(X_test.shape[1]):
        curveaxis[:,cx]=np.linspace(np.min(X_test[:,cx]),np.max(X_test[:,cx]),100)  
    curve_predictions = model.predict(curveaxis) 
    
    plt.subplot(5,3,i)
    plt.title(name)
    plt.scatter(X_test[:,0], Y_test,c='b')
    plt.scatter(X_test[:,0], Y_test_pred,c='r',alpha=0.5)
    plt.plot(curveaxis[:,0], curve_predictions,c='y')
    plt.grid()
    
    i=i+1 # subplot indeksi
# -----------------------------------------------------------------------------
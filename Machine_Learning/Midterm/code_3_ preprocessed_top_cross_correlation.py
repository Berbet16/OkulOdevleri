import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ------------------------QUESTION 1-------------------------------------------
# read data
df = pd.read_excel('C:/Users/BS/Desktop/MidtermReport_171805019_BetulBernaSoylu/RealEstateRegression.xlsx', usecols=('C:H'))
# -----------------------------------------------------------------------------


# ------------------------QUESTION 2-------------------------------------------
#regression normalize
preprocessing_data = pd.DataFrame(StandardScaler().fit(df).transform(df), columns=('X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude','Y house price of unit area'))
# -----------------------------------------------------------------------------


# ------------------------QUESTION 3-------------------------------------------
cross_cor_list=list()
cross_cor_and_colname=dict()
values_list=list()
max_=list()

# cross correlation calculation function
print("\nAbsoulute Cross-Correlation: \n")
def calculateAbsCrossCor(xname,i):
    x = np.array(preprocessing_data[xname])
    y = np.array(preprocessing_data["Y house price of unit area"])
    r=np.abs(np.corrcoef(x,y))
    cross_cor_list.append(r)
    cross_cor_and_colname[xname] = cross_cor_list[i][0][1]
    print(xname, "\n", r,"\n")
    
calculateAbsCrossCor("X2 house age",0) #column 1
calculateAbsCrossCor("X3 distance to the nearest MRT station",1) #column 2
calculateAbsCrossCor("X4 number of convenience stores",2)#column 3
calculateAbsCrossCor("X5 latitude",3)#column 4
calculateAbsCrossCor("X6 longitude",4)#column 5
# -----------------------------------------------------------------------------


# ------------------------QUESTION 4-------------------------------------------
#The function determining the highst cross correlation values
def Findmax(arr):
    sorted_cross_corr = sorted(arr.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    max1=sorted_cross_corr[0][1]
    max2=sorted_cross_corr[1][1]
    max1_name = sorted_cross_corr[0][0]
    max2_name = sorted_cross_corr[1][0]
    max_.append(max1)
    max_.append(max2)
    print("First Max Value:",max1, "\nSecond Max Value:",max2)
    return max1_name, max2_name

max1_name, max2_name = Findmax(cross_cor_and_colname)

#creates a new dataset with high cross correlation values 
max_feature_df = pd.DataFrame()
max_feature_df[max1_name] = preprocessing_data[max1_name]
max_feature_df[max2_name] = preprocessing_data[max2_name]
max_feature_df["Y house price of unit area"] = preprocessing_data["Y house price of unit area"]

X=max_feature_df.iloc[:,:-1].values
y=max_feature_df['Y house price of unit area'].values
# -----------------------------------------------------------------------------


# ------------------------QUESTION 5-------------------------------------------
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


# ------------------------QUESTION 6-------------------------------------------
#separates data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#calculate r2 and mse value
def scoreResults(model, X_train, X_test, y_train, y_test):
    
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    r2_train = metrics.r2_score(y_train, y_train_predict)
    r2_test = metrics.r2_score(y_test, y_test_predict)

    mse_train = metrics.mean_squared_error(y_train, y_train_predict)
    mse_test = metrics.mean_squared_error(y_test, y_test_predict)

    return [r2_train, r2_test, mse_train, mse_test, y_train_predict, y_test_predict]

#  makes 10-fold process
def regressionAlgrithm(reg_model, X_train, X_test, y_train, y_test, reg_name):
    
    model = reg_model
    k = 10
    iter = 1
    cv = KFold(n_splits=k, random_state = 0, shuffle=True)
    print(reg_name, "Scores")
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        model.fit(X_train, y_train)
        
        result = scoreResults(model = model
                          ,X_train = X_train
                          ,X_test = X_test
                          ,y_train = y_train
                          ,y_test = y_test)
        
        print(f"{iter}. veri kesiti")
        print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
        print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
        iter += 1
        
    print("------------------------------------------------------------------------")

regressionAlgrithm(lr, X_train, X_test, y_train, y_test, 'Linear Regression Algorithm')
regressionAlgrithm(ridge_model, X_train, X_test, y_train, y_test, 'Ridge Regression Algorithm')
regressionAlgrithm(lasso_model, X_train, X_test, y_train, y_test, 'Lasso Regression Algorithm')
regressionAlgrithm(knn_model, X_train, X_test, y_train, y_test, 'KNN Regression Algorithm')
regressionAlgrithm(elastic_model, X_train, X_test, y_train, y_test, 'Elastic Net Regression Algorithm')
regressionAlgrithm(decision_model, X_train, X_test, y_train, y_test, 'Decision Tree Regression Algorithm')
regressionAlgrithm(random_model, X_train, X_test, y_train, y_test, 'Random Forest Regression Algorithm')
regressionAlgrithm(support_model, X_train, X_test, y_train, y_test, 'Super Vector Regression Algorithm')

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
    
    resultsMSE = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    resultsR2 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    
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

# plotting histogram of MSE and R2
r2_hist = plt.hist(valuesR2)
plt.title('Histogram of R2 results')
plt.show()

mse_hist = plt.hist(valuesMSE)
plt.title('Histogram of MSE results')
plt.show()
# -----------------------------------------------------------------------------


# ------------------------QUESTION 7-------------------------------------------
# residu and curve plot
X=preprocessing_data.iloc[:,:-1].values
Y=preprocessing_data['Y house price of unit area'].values

# separates data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

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
    
    '''print(name,":", mean_squared_error(Y_test, Y_test_pred))'''
    plt.subplot(5,3,i)
    plt.title(name)
    plt.scatter(X_test[:,0], Y_test,c='b')
    plt.scatter(X_test[:,0], Y_test_pred,c='r',alpha=0.5)
    plt.plot(curveaxis[:,0], curve_predictions,c='y')
    plt.grid()
    
    i=i+1 # subplot indeksi
# -----------------------------------------------------------------------------
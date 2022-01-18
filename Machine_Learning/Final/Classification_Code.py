import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import operator


# ********************* Read Data **************************************************************
df = pd.read_excel('C:/Users/BS/Desktop/Yapay Öğrenme/FINAL/metadata_train.xls', header = None)
df.drop([0], axis = 0, inplace = True)

X = df.iloc[:,:-1].values
y = df[3].values.astype('int')
# **********************************************************************************************

# ********************* Train & Test ***********************************************************
#separates data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# **********************************************************************************************

# ********************* Preprocessing (QUESTION 2) *********************************************
preprocessing_data = pd.DataFrame(StandardScaler().fit(df).transform(df))
# **********************************************************************************************

# ********************* Feature Extraction (QUESTION 2) ****************************************
def getfeature(data):
    fmean=np.mean(data)
    fstd=np.std(data)
    fmax=np.max(data)
    fmin=np.min(data)
    fkurtosis=scipy.stats.kurtosis(data)
    zero_crosses = np.nonzero(np.diff(data > 0))[0]
    fzero=zero_crosses.size/len(data)
    return fmean,fstd,fmax,fmin,fkurtosis,fzero
def extractFeature(raw_data,ws,hop,dfname):
    fmean=[]
    fstd=[]
    fmax=[]
    fmin=[]
    fkurtosis=[]
    fzero=[]
    flabel=[]
    for i in range(ws,len(raw_data),hop):
       m,s,ma,mi,k,z = getfeature(raw_data.iloc[i-ws+1:i,0])
       fmean.append(m)
       fstd.append(s)
       fmax.append(ma)
       fmin.append(mi)
       fzero.append(z)
       fkurtosis.append(k)
       
       flabel.append(dfname)
    rdf = pd.DataFrame(
    {'mean': fmean,
     'std': fstd,
     'max': fmax,
     'min': fmin,
     'kurtosis': fkurtosis,
     'zerocross':fzero,
     'label':flabel
    })
    return rdf
# not fault train
notfault = extractFeature(preprocessing_data,250,10,"0")
# fault train
fault = extractFeature(preprocessing_data,250,10,"1")

merged = pd.concat([notfault, fault])
# **************************************************************************************


# ************************ Classification Algorithms (QUESTION 3) **********************
classifier_dt = DecisionTreeClassifier()
classifier_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
classifier_lr = LogisticRegression(random_state = 0)
classifier_nb = GaussianNB()  
classifier_rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0) 
classifier_ab = AdaBoostClassifier(n_estimators=50, learning_rate=1)
classifier_sv = SVC()
classifier_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

def classificationAlgorithms(cls_name, cls_model, X_train, X_test, y_train, y_test):
    model = cls_model
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)


    print(cls_name, 'accuracy score', accuracy_score(y_true = y_train, y_pred = model.predict(X_train)))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred), '\n')
    

classificationAlgorithms('Decision Tree Algorithm', classifier_dt, X_train, X_test, y_train, y_test)
classificationAlgorithms('KNN Algorithm', classifier_knn, X_train, X_test, y_train, y_test)
classificationAlgorithms('Logistic Regression Algorithm', classifier_lr, X_train, X_test, y_train, y_test)
classificationAlgorithms('Naive Bayes Algorithm', classifier_nb, X_train, X_test, y_train, y_test)
classificationAlgorithms('Random Forest Algorithm', classifier_rf, X_train, X_test, y_train, y_test)
classificationAlgorithms('AdaBoost Algorithm', classifier_ab, X_train, X_test, y_train, y_test)
classificationAlgorithms('SVC Algorithm', classifier_sv, X_train, X_test, y_train, y_test)
classificationAlgorithms('MLP Algorithm', classifier_mlp, X_train, X_test, y_train, y_test)
# **************************************************************************************

# ************************ K-FOLD & BOXPLOTS (QUESTION 4) ******************************
models = []

models.append(classifier_dt)
models.append(classifier_knn)
models.append(classifier_lr)
models.append(classifier_nb)
models.append(classifier_rf)
models.append(classifier_ab)
models.append(classifier_sv)
models.append(classifier_mlp)

names = []

names.append('DT')
names.append('KNN')
names.append('LR')
names.append('NB')
names.append('RF')
names.append('AB')
names.append('SVC')
names.append('MLP')


results = []

for model in models:
    score = cross_val_score(model, X_train, y_train, cv=10)
    results.append(score)
    
puan = []

for i in range(len(names)):
    puan.append(results[i].mean())
print("Highest accuracy value:")
print(names[puan.index(max(puan))], max(puan))  


ax = sns.boxplot(data = results)
ax.set_xticklabels(names)
plt.show()
#****************************************************************************************

#***************************HYPERPARAMETERES (QUESTION 5)********************************
def dicbar(title, dic):
    keys = dic.keys()
    values = dic.values()
    plt.xticks(rotation=90)
    plt.title(title)
    plt.bar(keys, values)
    plt.show()
    
DecisionTreeClassifierHP = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [1, 2, 3, 4, 5, 6],
    "min_samples_split": [3, 4, 5, 6, 7, 8, 9],
    "min_samples_leaf": [1, 2, 3, 4, 5, 6],
    "max_features": ["auto", "sqrt", "log2", None]
}
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Create the parameters list we wish to tune.
parameters = DecisionTreeClassifierHP


# Perform grid search on the classifier using 'scores' as the scoring method.
# Create the object.
grid_obj = GridSearchCV(classifier_dt, parameters, cv=20, scoring = 'accuracy')

# Fit the grid search object to the training data and find the optimal parameters.
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator.
best_clf = grid_fit.best_estimator_

# Fit the new model.
best_clf.fit(X_train, y_train)

# Make predictions using the new model.
best_train_predictions = best_clf.predict(X_train)
best_test_predictions = best_clf.predict(X_test)

# Plot the new model.
#plot_model(X, y, best_clf)

# Let's also explore what parameters ended up being used in the new model.
best_clf

grid_fit.best_score_

hyperparameters = {}
for i in range(len(grid_fit.cv_results_["mean_test_score"])):
    hyperparameters[str(grid_fit.cv_results_["params"][i])] = grid_fit.cv_results_["mean_test_score"][i]
    #print(str(grid_fit.cv_results_["params"][i]) +": "+ str(grid_fit.cv_results_["mean_test_score"][i]))
    
keys = hyperparameters.keys()
values = hyperparameters.values()
plt.xticks(rotation=90)
plt.bar(keys,values)
plt.show()

hyperparameters = sorted(hyperparameters.items(), key=lambda x: -x[1])

from sklearn.metrics import classification_report

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    classifier_dt = GridSearchCV(DecisionTreeClassifier(), parameters, scoring='%s_macro' % score)
    classifier_dt.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(classifier_dt.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = classifier_dt.cv_results_['mean_test_score']
    stds = classifier_dt.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier_dt.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, classifier_dt.predict(X_test)
    print(classification_report(y_true, y_pred))
    
    print('Best Score: %s' % classifier_dt.best_score_)
    print('Best Hyperparameters: %s' % classifier_dt.best_params_)

    predictions = classifier_dt.predict(X_test)
    
    conf = confusion_matrix(y_test, predictions)
    plt.figure()
    sns.heatmap(conf, annot=True)
    plt.xlabel("True Label")
    plt.ylabel("Prediction Label")
#****************************************************************************************

#********************* HYper-Parameter (QUESTION 6) *************************************
sortedHP  = {}
sortedHP = sorted(hyperparameters.items(),key = operator.itemgetter(1),reverse=True)
print(sortedHP)

dictHP = {}
for hyperparameters in sortedHP[0:3]:
    dictHP[hyperparameters[0]] = hyperparameters[1]
    print(hyperparameters[1])
    
plt.xsticks(rotation=90)
plt.hist(dictHP)

#****************************************************************************************

#*********************** Plot Predictions (QUESTION 7) **********************************
predictions = []
predictions.append(classifier_dt.predict([[12, 0]])[0])
predictions.append(classifier_dt.predict([[1, 1]])[0])
predictions.append(classifier_dt.predict([[5, 2]])[0])

#plot draw
plt.bar(np.arange(len(predictions)), predictions)
#****************************************************************************************












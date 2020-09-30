import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

###-------------------Create the data frame-------------------------------###

data=pd.read_csv('C:/Users/shoha/Desktop/ML_part2/Xy_train.csv') #read the excel file from location on pc
results=pd.read_csv('C:/Users/shoha/Desktop/ML_part2/results.csv') #read the excel file from location on pc
results_ann=pd.read_csv('C:/Users/shoha/Desktop/ML_part2/results_ann.csv') #read the excel file from location on pc

###-------------------Change invalid values -------------------------------###

np.random.seed(356) #seed from normal distribution
age120=data[data['age']<120] #subset age under 120
agemean =np.round_(np.mean(age120['age'])) #mean to all age under 120
agestd = np.std(age120['age']) #std to all age under 120
for i in data.index: #changing ages over 120 to random number fro normal distribution with agemean, agestd
       if (data['age'].values[i] > 120):
           data['age'].values[i] = np.round_(np.random.normal(agemean, agestd, 1))

###-------------------Changing missing values -------------------------------###

#CA
data.loc[data['ca'] == 4,'ca'] = 0

#Thal
data.loc[data['thal'] == 0,'thal'] = 2

###-------------------Changing to dummies -------------------------------###

#Chol changing to categorical variable
data.loc[data['chol'] <= 200, 'chol'] = 0  #change chol to 0\1 , 0= normal 1=high
data.loc[data['chol'] > 300, 'chol'] = 2  #change chol to 0\1 , 0= normal 1=high
data.loc[data['chol'] > 200,'chol'] = 1  #change chol to 0\1 , 0= normal 1=high

dataDummies = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'ca', 'thal' , 'chol'], drop_first=True)
dataDummies=dataDummies.drop(['id'], axis=1)

for col in dataDummies.columns:
    print(col)

###---------------------------- Validation set & Train set  -----------------------------###

X_train = dataDummies.drop(['y'], axis=1).values
y_train = dataDummies['y'].values
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

print("val")
count=0
for i in np.arange(0,len(y_val)):
    if (y_val[i]==1):
        count=count+1
print(count)
print(len(y_val))

print("train")
count2=0
for i in np.arange(0,len(y_train)):
    if (y_train[i]==1):
        count2=count2+1
print(count2)
print(len(y_train))

###---------------------------- Kfold 9 - Full Max Depth Tree -----------------------------###

kfold = KFold(n_splits=9, shuffle=True, random_state=123)

DT_res = pd.DataFrame()
for train_idx, val_idx in kfold.split(X_train):
       modelDT = DecisionTreeClassifier(criterion='entropy', random_state=123)
       modelDT.fit(X_train[train_idx], y_train[train_idx])
       accTrain=accuracy_score(y_true=y_train[train_idx], y_pred=modelDT.predict(X_train[train_idx]))
       accVal = accuracy_score(y_train[val_idx], modelDT.predict(X_train[val_idx]))
       DT_res = DT_res.append({'accVal': accVal , 'accTrain' : accTrain}, ignore_index=True)

print("Max Depth Tree Performances:")
print(round(DT_res,3))
print(round(DT_res.mean(),3))

preds_DT = modelDT.predict(X_val)
print("Max Depth Tree- Validation accuracy: ", round(accuracy_score(y_val, preds_DT), 3))
print()

###--------------------------------------Hyperparameter tuning - Tree-------------------------------###

DTGrid= {'max_depth' : np.arange(1, 11, 1),
         'criterion' : ['gini', 'entropy'],
         'splitter' : ['best', 'random']}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=DTGrid, refit=True, cv=9)
grid_search.fit(X_train, y_train)

#Show Results
#results_dt=pd.DataFrame(grid_search.cv_results_)
#results_dt=results_dt.drop(['mean_fit_time','std_fit_time','mean_score_time','std_score_time','rank_test_score'],1)
#results_dt.to_csv("C:/Users/shoha/Desktop/results.csv")

print("Best DT Hyper Parameters: ", grid_search.best_params_)
print("Best DT Score: ", round(grid_search.best_score_,3))

best_model = grid_search.best_estimator_
preds_DT = best_model.predict(X_val)
print("DT- Validation accuracy: ", round(accuracy_score(y_val, preds_DT), 3))

DTGrid_df = pd.DataFrame(columns=['index','max_depth', 'criterion', 'splitter', 'train ac', 'val ac'])
criterion_list = {'gini', 'entropy'}
splitter_list={'best', 'random'}

print("confusion matrix DT:")
print(confusion_matrix(y_true=y_val, y_pred=preds_DT))


for j in np.arange(0,40,1):
    model_csi = DecisionTreeClassifier(criterion=results['param_criterion'][j], splitter=results['param_splitter'][j], max_depth=results['param_max_depth'][j], random_state=42)
    model_csi.fit(X_train, y_train)
    preds_csi = model_csi.predict(X_val)
    DTGrid_df.at[j, 'index'] = j
    DTGrid_df.at[j, 'max_depth'] = results['param_max_depth'][j]
    DTGrid_df.at[j, 'criterion'] = results['param_criterion'][j]
    DTGrid_df.at[j, 'splitter'] = results['param_splitter'][j]
    DTGrid_df.at[j, 'train ac'] = round(results['mean_test_score'][j], 3)
    DTGrid_df.at[j, 'val ac'] = round(results['val ac'][j], 3)
    #DTGrid_df.at[j, 'val ac'] = round(accuracy_score(y_val, preds_csi), 3)
    #results.at['val ac'][j]= round(accuracy_score(y_val, preds_csi), 3)

#results.to_csv("C:/Users/shoha/Desktop/results.csv")
print(DTGrid_df)
plt.figure(figsize=(13, 4))
plt.plot(DTGrid_df['index'], DTGrid_df['train ac'], marker='o', markersize=4)
plt.plot(DTGrid_df['index'], DTGrid_df['val ac'], marker='o', markersize=4)
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.title('Accuracy percentages on train set  and validation set for each model')
plt.xlabel('Table Index')
plt.ylabel('Accuracy Percentages')
plt.show()

###---------------------------------------------------print trees--------------------------------------------------------------##

plt.figure(figsize=(20, 10))
plot_tree(best_model, filled=True, max_depth=3,class_names=['no Heart attak', 'Heart attak'], feature_names=['age', 'gender', 'trestbps','fbs','thalach','exang' ,
                                                                                    'oldpeak','cp_1','cp_2','cp_3','restecg_1','restecg_2',
                                                                                    'slope_1','slope_2','ca_1','ca_2','ca_3', 'thal_2','thal_3',
                                                                                    'chol_1','chol_2'])
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(best_model, filled=True, class_names=['no Heart attak', 'Heart attak'], feature_names=['age', 'gender', 'trestbps','fbs','thalach','exang' ,
                                                                                    'oldpeak','cp_1','cp_2','cp_3','restecg_1','restecg_2',
                                                                                    'slope_1','slope_2','ca_1','ca_2','ca_3', 'thal_2','thal_3',
                                                                                    'chol_1','chol_2'])
plt.show()

###---------------------------------------------------features importances---------------------------------------------##

print("features:")
print(best_model.n_features_)

print('features importances:')
print(best_model.feature_importances_)

###---------------------------------------------------ANN-scale values--------------------------------------------------------------##

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)

###---------------------------------------------------ANN-default network--------------------------------------------------------------##

ANNres = pd.DataFrame()
for train_idx, val_idx in kfold.split(X_train_scaled):
      modelANN= MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
             beta_2=0.999, early_stopping=False, epsilon=1e-08,
             hidden_layer_sizes=(100,) , learning_rate='constant',
             learning_rate_init=0.001, max_fun=15000, max_iter=200,
             momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
             power_t=0.5, random_state=None, shuffle=True, solver='adam',
             tol=0.0001, validation_fraction=0.1, verbose=False,
             warm_start=False)
      modelANN.fit(X_train_scaled[train_idx], y_train[train_idx])
      accTrain2=accuracy_score(y_true=y_train[train_idx], y_pred=modelANN.predict(X_train_scaled[train_idx]))
      accVal2 = accuracy_score(y_train[val_idx], modelANN.predict(X_train_scaled[val_idx]))
      ANNres = ANNres.append({'accVal': accVal2 , 'accTrain' : accTrain2}, ignore_index=True)

print("Default ANN Performances:")
print(round(ANNres,3))
print(round(ANNres.mean(),3))

preds_ANN = modelANN.predict(X_val_scaled)
print("Default ANN- Validation accuracy: ", round(accuracy_score(y_val, preds_ANN), 3))
print()

###---------------------------------------------------ANN-Best network--------------------------------------------------------------##

AnnGrid= {'activation' : ['logistic', 'tanh', 'relu'],
        'max_iter' : np.arange(100,1500,25),
        'learning_rate_init' : np.arange(0.001 ,0.050, 0.005),
        'solver' : ['lbfgs' , 'sgd' ],
        'hidden_layer_sizes' : [(3,),(3,3,),(3,3,3,),(3,3,3,3,),(4,),(4,4,),(4,4,4,),(4,4,4,4,),(5,),(5,5,),(5,5,5,),(5,5,5,5,),
                                (6,6,6,6,),(6,6,6,),(6,6,),(6,),(7,),(7,7,),(7,7,7,),(7,7,7,7,),(8,),(8,8,),(8,8,8,),(8,8,8,8,),
                                (9,),(9,9,),(9,9,9,),(9,9,9,9,)]}


random_search = RandomizedSearchCV(MLPClassifier(random_state=42), param_distributions=AnnGrid, cv=9, random_state=123, n_iter=300, refit=True)
random_search.fit(X_train_scaled, y_train)
print("Best ANN Hyper Parameters: ", random_search.best_params_)
print("Best ANN Score: ", round(random_search.best_score_,3))

best_model_ANN = random_search.best_estimator_
preds_ANN = best_model_ANN.predict(X_val_scaled)
print("ANN- Validation accuracy: ", round(accuracy_score(y_val, preds_ANN), 3))
print("confusion matrix ANN:")
print(confusion_matrix(y_true=y_val, y_pred=preds_ANN))

#Show.Results
#results_ann=pd.DataFrame(random_search.cv_results_)
#results_ann=results_ann.drop(['mean_fit_time','std_fit_time','mean_score_time','std_score_time','rank_test_score'],1)
#results_ann.to_csv('C:/Users/shoha/Desktop/ML_part2/results_ann.csv')


ANNGrid_df = pd.DataFrame(columns=['index','activation', 'max_iter', 'learning_rate_init', 'solver', 'hidden_layer_sizes','train ac', 'val ac'])
activation_list = {'idntity', 'logistic', 'tanh', 'relu'}
max_iter_list= np.arange(100,1500,25),
learning_rate_init_list=np.arange(0.001 ,0.050, 0.005)
solver_list = {'lbfgs' , 'sgd' , 'adam'}
hidden_layer_sizes_list= {(3,),(3,3,),(3,3,3,),(3,3,3,3,),(4,),(4,4,),(4,4,4,),(4,4,4,4,),(5,),(5,5,),(5,5,5,),(5,5,5,5,),
                                (6,6,6,6,),(6,6,6,),(6,6,),(6,),(7,),(7,7,),(7,7,7,),(7,7,7,7,),(8,),(8,8,),(8,8,8,),(8,8,8,8,),
                                (9,),(9,9,),(9,9,9,),(9,9,9,9,)}

for j in np.arange(0,300,1):
    str = results_ann['param_hidden_layer_sizes'][j]
    s2 = ","
    number = str[1]
            # + str[2]
    numberof = str.count(s2)
    if (numberof == 1):
        x = (int(number),)
    if (numberof == 2):
        x = (int(number), int(number),)
    if (numberof == 3):
        x = (int(number), int(number), int(number),)
    if (numberof == 4):
        x = (int(number), int(number), int(number), int(number),)

    model_i = MLPClassifier(activation=results_ann['param_activation'][j], max_iter=results_ann['param_max_iter'][j],
                           learning_rate_init=results_ann['param_learning_rate_init'][j], solver=results_ann['param_solver'][j],
                           hidden_layer_sizes=x, random_state=42)
    model_i.fit(X_train_scaled,y_train)
    preds_i = model_i.predict(X_val_scaled)
    ANNGrid_df.at[j, 'index'] = j
    ANNGrid_df.at[j, 'activation'] = results_ann['param_activation'][j]
    ANNGrid_df.at[j, 'max_iter'] = results_ann['param_max_iter'][j]
    ANNGrid_df.at[j, 'learning_rate_init'] = results_ann['param_learning_rate_init'][j]
    ANNGrid_df.at[j, 'solver'] = results_ann['param_solver'][j]
    ANNGrid_df.at[j, 'hidden_layer_sizes'] = results_ann['param_hidden_layer_sizes'][j]
    ANNGrid_df.at[j, 'train ac'] = round(results_ann['mean_test_score'][j], 3)
    #ANNGrid_df.at[j, 'val ac'] = round(results_ann['val ac'][j], 3)
    ANNGrid_df.at[j, 'val ac'] = round(accuracy_score(y_val, preds_i), 3)
    results_ann.at[j,'val ac']= round(accuracy_score(y_val, preds_i), 3)

#results_ann.to_csv('C:/Users/shoha/Desktop/ML_part2/results_ann.csv')

print(ANNGrid_df)
plt.figure(figsize=(13, 4))
plt.plot(ANNGrid_df['index'], ANNGrid_df['train ac'], marker='o', markersize=4)
plt.plot(ANNGrid_df['index'], ANNGrid_df['val ac'], marker='o', markersize=4)
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.title('Accuracy percentages on train set  and validation set for each model')
plt.xlabel('Table Index')
plt.ylabel('Accuracy Percentages')
plt.show()

###--------------------------------------------------- PCA --------------------------------------------------------------##

pca = PCA(n_components=2)
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_train_pca = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2'])

pca.fit(X_val_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_val_pca = pd.DataFrame(X_val_pca, columns=['PC1', 'PC2'])

X_train_pca['y'] = y_train
sns.scatterplot(x='PC1', y='PC2', hue='y', data=X_train_pca)
plt.title('Train set after PCA')
plt.show()

X_val_pca['y'] = y_val
sns.scatterplot(x='PC1', y='PC2', hue='y', data=X_val_pca)
plt.title('Validation set after PCA')
plt.show()

###--------------------------------------------------- KMeans- default clustring ------------------------------------##

X_train_pca = X_train_pca.drop(['y'], axis=1)
X_val_pca = X_val_pca.drop(['y'], axis=1)

kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                precompute_distances='deprecated', verbose=0, random_state=None,
                copy_x=True, n_jobs='deprecated', algorithm='auto')
modelK = kmeans.fit(X_train_pca)

pred_train=modelK.predict(X_train_pca.values)
for i in np.arange(0,len(pred_train),1):
    if(pred_train[i]==1):
        pred_train[i]=0
    else: pred_train[i]=1
print("Default KMeans- train accuracy: ",round(accuracy_score(y_true=y_train,y_pred=pred_train),3))
print("confusion matrix- train set:")
print(confusion_matrix(y_true=y_train, y_pred=pred_train))

pred_val=modelK.predict(X_val_pca.values)
for i in np.arange(0,len(pred_val),1):
    if(pred_val[i]==1):
        pred_val[i]=0
    else: pred_val[i]=1
print("Default KMeans- Validation accuracy: ",round(accuracy_score(y_true=y_val,y_pred=pred_val),3))
print("confusion matrix- validation set:")
print(confusion_matrix(y_true=y_val, y_pred=pred_val))

centers=modelK.cluster_centers_
print(centers)
print(round(centers[0][0],3))
print(round(centers[0][1],3))
print(round(centers[1][0],3))
print(round(centers[1][1],3))

x_test = np.linspace(-4, 4, 100)
y_test = np.linspace(-4, 7, 100)
predictions = pd.DataFrame()
for x in tqdm(x_test):
    for y in y_test:
        pred = modelK.predict(np.array([x, y]).reshape(-1, 2))[0]
        predictions = predictions.append(dict(X1=x, X2=y, y=pred), ignore_index=True)

plt.figure(figsize=(7,7))
plt.scatter(x=predictions[predictions.y == 1]['X1'], y = predictions[predictions.y == 1]['X2'], c='powderblue')
plt.scatter(x=predictions[predictions.y == 0]['X1'], y = predictions[predictions.y == 0]['X2'], c='ivory')
X_train_pca['y'] = y_train
sns.scatterplot(x='PC1', y='PC2', hue='y', data=X_train_pca)
plt.scatter(centers[:, 0], centers[:, 1], marker='+', s=100 ,color='red')
plt.title("Clustering with K-Means- train set")
plt.show()

plt.figure(figsize=(7,7))
plt.scatter(x=predictions[predictions.y == 1]['X1'], y = predictions[predictions.y == 1]['X2'], c='powderblue')
plt.scatter(x=predictions[predictions.y == 0]['X1'], y = predictions[predictions.y == 0]['X2'], c='ivory')
X_val_pca['y'] = y_val
sns.scatterplot(x='PC1', y='PC2', hue='y', data=X_val_pca)
plt.scatter(centers[:, 0], centers[:, 1], marker='+', s=100 ,color='red')
plt.title("Clustering with K-Means- validation set")
plt.show()

###--------------------------------------------------- KMeans- K clustres ------------------------------------##

X_train_pca = X_train_pca.drop(['y'], axis=1)
X_val_pca = X_val_pca.drop(['y'], axis=1)

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
for n_clusters in range(2, 10, 1):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_train_pca)
    assignments = kmeans.predict(X_train_pca)
    scheme = X_train_pca.copy()
    scheme = pd.DataFrame(scheme, columns=['PC1', 'PC2'])
    scheme['cluster'] = assignments
    i = 0 if n_clusters in [2, 3, 4, 5] else 1
    j = 0
    j = 1 if n_clusters in [3, 7] else j
    j = 2 if n_clusters in [4, 8] else j
    j = 3 if n_clusters in [5, 9] else j
    sns.scatterplot(x='PC1', y='PC2', data=scheme, hue='cluster', ax=axes[i, j], palette='Accent_r', legend=False)
plt.show()

iner_list = []
dbi_list = []
sil_list = []
ch_list = []

for n_clusters in tqdm(range(2, 10, 1)):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_train_pca)
    assignment = kmeans.predict(X_train_pca)
    iner = kmeans.inertia_
    sil = silhouette_score(X_train_pca, assignment)
    dbi = davies_bouldin_score(X_train_pca, assignment)
    ch = calinski_harabasz_score(X_train_pca, assignment)
    dbi_list.append(dbi)
    sil_list.append(sil)
    ch_list.append(ch)
    iner_list.append(iner)

plt.plot(range(2, 10, 1), iner_list, marker='o')
plt.title("Inertia")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, 10, 1), sil_list, marker='o')
plt.title("Silhouette Coefficient")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, 10, 1), dbi_list, marker='o')
plt.title("Davies-Bouldin Index")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, 10, 1), ch_list, marker='o')
plt.title("Calinski-Harabasz Index")
plt.xlabel("Number of clusters")
plt.show()

###--------------------------------------------------- SVM ------------------------------------##

print("svm:")
SVMGrid= {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 'C' : np.arange(0.05, 3.5, 0.5), 'gamma' : ['scale', 'auto']}

grid_searchSVM = GridSearchCV(estimator=SVC(random_state=42), param_grid=SVMGrid, refit=True, cv=9)
grid_searchSVM.fit(X_train_pca, y_train)
print("Best SVM Hyper Parameters: ", grid_searchSVM.best_params_)
print("Best SVM Score: ", round(grid_searchSVM.best_score_,3))

best_modelSVM = grid_searchSVM.best_estimator_
preds_SVM = best_modelSVM.predict(X_val_pca)
print("before")
print("SVM- Validation accuracy: ", round(accuracy_score(y_val, preds_SVM), 3))
print(confusion_matrix(y_true=y_val, y_pred=preds_SVM))

for i in np.arange(0,len(preds_SVM),1):
    if(preds_SVM[i]==1):
        preds_SVM[i]=0
    else: preds_SVM[i]=1

print("SVM- Validation accuracy: ", round(accuracy_score(y_val, preds_SVM), 3))
print("after")
print(confusion_matrix(y_true=y_val, y_pred=preds_SVM))

x_test = np.linspace(-5, 5, 100)
y_test = np.linspace(-5, 5, 100)
predictions = pd.DataFrame()
for x in x_test:
    for y in y_test:
        pred = best_modelSVM.predict(np.array([x, y]).reshape(-1, 2))[0]
        predictions = predictions.append(dict(X1=x, X2=y, y=pred), ignore_index=True)

plt.scatter(x=predictions[predictions.y == 0]['X1'], y = predictions[predictions.y == 0]['X2'], c='powderblue')
plt.scatter(x=predictions[predictions.y == 1]['X1'], y = predictions[predictions.y == 1]['X2'], c='ivory')
X_train_pca['y']=y_train
sns.scatterplot(x='PC1', y='PC2', data=X_train_pca, hue='y')
plt.title("Clustering with SVM- train set")
plt.show()


plt.scatter(x=predictions[predictions.y == 0]['X1'], y = predictions[predictions.y == 0]['X2'], c='powderblue')
plt.scatter(x=predictions[predictions.y == 1]['X1'], y = predictions[predictions.y == 1]['X2'], c='ivory')
X_val_pca['y']=y_val
sns.scatterplot(x='PC1', y='PC2', data=X_val_pca, hue='y')
plt.title("Clustering with SVM- validation set")
plt.show()

X_train_pca=X_train_pca.drop(['y'],1)
X_train_pca['cluster'] = best_modelSVM.predict(X_train_pca.values)
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_train_pca, palette='Accent')
plt.title('SVM clustering on train set')
plt.show()

###--------------------------------------------------- Final Model Predictions ------------------------------------##

X_test=pd.read_csv('C:/Users/shoha/Desktop/ML_part2/X_test.csv') #read the excel file from location on pc
y_pred = pd.DataFrame(columns=['y'])

#Change invalid values
np.random.seed(356) #seed from normal distribution
age120=data[data['age']<120] #subset age under 120
agemean =np.round_(np.mean(age120['age'])) #mean to all age under 120
agestd = np.std(age120['age']) #std to all age under 120
for i in X_test.index: #changing ages over 120 to random number fro normal distribution with agemean, agestd
       if (X_test['age'].values[i] > 120):
           X_test['age'].values[i] = np.round_(np.random.normal(agemean, agestd, 1))

#Changing missing values
#CA
X_test.loc[X_test['ca'] == 4,'ca'] = 0
#Thal
X_test.loc[X_test['thal'] == 0,'thal'] = 2

#Changing to dummies
#Chol changing to categorical variable
X_test.loc[X_test['chol'] <= 200, 'chol'] = 0  #change chol to 0\1 , 0= normal 1=high
X_test.loc[X_test['chol'] > 300, 'chol'] = 2  #change chol to 0\1 , 0= normal 1=high
X_test.loc[X_test['chol'] > 200,'chol'] = 1  #change chol to 0\1 , 0= normal 1=high

X_test_dum = pd.get_dummies(X_test, columns=['cp', 'restecg', 'slope', 'ca', 'thal' , 'chol'], drop_first=True)
X_test_dum=X_test_dum.drop(['id'], axis=1)

y_pred['y']= best_model.predict(X_test_dum.values)
y_pred.to_csv('C:/Users/shoha/Desktop/ML_part2/y_test.csv')

###--------------------------------------------------- Random Forest - Bonus Question ------------------------------------##

ramdom_forest_grid= { 'n_estimators': np.arange(10,110,10), 'criterion' : ['gini', 'entropy'], 'max_depth': np.arange(1,11,1), 'max_features' :['sqrt', 'log2']}
grid_search_rf = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42), param_distributions=ramdom_forest_grid,  cv=9, random_state=123, n_iter=100, refit=True)
grid_search_rf.fit(X_train,y_train)

print("Best RF Hyper Parameters: ", grid_search_rf.best_params_)
print("Best RF Score: ", round(grid_search_rf.best_score_,3))

best_model_rf = grid_search_rf.best_estimator_
preds_rf = best_model_rf.predict(X_val)
print("RF- Validation accuracy: ", round(accuracy_score(y_val, preds_rf), 3))

preds_rf = best_model_rf.predict(X_val)
print("confusion matrix RF:")
print(confusion_matrix(y_true=y_val, y_pred=preds_rf))

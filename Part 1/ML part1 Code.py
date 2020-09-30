
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from matplotlib import ticker
from statsmodels.graphics.mosaicplot import mosaic

###-------------------Create the data frame-------------------------------###

data=pd.read_csv('A:\אוניברסיטה\שנה ג\סמסטר ב\משין\פרויקט\Xy_train.csv') #read the excel file from location on pc

###-------------------Change invalid values -------------------------------###

np.random.seed(356) #seed from normal distribution
age120=data[data['age']<120] #subset age under 120
agemean =np.round_(np.mean(age120['age'])) #mean to all age under 120
agestd = np.std(age120['age']) #std to all age under 120
for i in data.index: #changing ages over 120 to random number fro normal distribution with agemean, agestd
        if (data['age'].values[i] > 120):
            data['age'].values[i] = np.round_(np.random.normal(agemean, agestd, 1))
data=data.drop(['Unnamed: 0' , 'newAge'], axis=1)

###-------------------Prior Probabilities and histograms-------------------###

#hist of y
plt.figure(figsize=(5,5))
Yhist=sns.countplot(x=data['y'], data=data)
total = len(data['y'])
plt.title('heart attack prior probabilities')
plt.xlabel('Category')
plt.ylabel('Count')
plt.ylim(0,150)
textstr = '\n'.join((
   r'0 = no heart attack',
   r'1 = heart attack'))
for p in Yhist.patches:
      percentage = '{:.1f}%'.format(100 * p.get_height()/total)
      x = p.get_x() + p.get_width()/2
      y = p.get_y() + p.get_height() -0.02
      Yhist.annotate(percentage, (x, y))
Yhist.yaxis.set_major_locator(ticker.LinearLocator(11))
Yhist.text(0.65, 0.95, textstr, transform=Yhist.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='left')
plt.show()

#hist of gender
plt.figure(figsize=(5,5))
Genderhist=sns.countplot(x=data['gender'], data=data)
plt.title('gender prior probabilities')
plt.xlabel('Category')
plt.ylabel('Count')
plt.ylim(0,180)
textstr = '\n'.join((
   r'0 = female',
   r'1 = male'))
for p in Genderhist.patches:
      percentage = '{:.1f}%'.format(100 * p.get_height()/total)
      x = p.get_x() + p.get_width()/2
      y = p.get_y() + p.get_height() -0.02
      Genderhist.annotate(percentage, (x, y))
Genderhist.yaxis.set_major_locator(ticker.LinearLocator(11))
Genderhist.text(0.65, 0.95, textstr, transform=Genderhist.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='left')
plt.show()

#hist of cp
plt.figure(figsize=(5,5))
CPhist=sns.countplot(x=data['cp'], data=data)
plt.title('cp prior probabilities')
plt.xlabel('Category')
plt.ylabel('Count')
plt.ylim(0,110)
textstr = '\n'.join((
   r'0 = typical angina',
   r'1 = atypical angina',
   r'2 = non-anginal pain',
   r'3 = asymptomatic'))
for p in CPhist.patches:
      percentage = '{:.1f}%'.format(100 * p.get_height()/total)
      x = p.get_x() + p.get_width()/2
      y = p.get_y() + p.get_height() -0.02
      CPhist.annotate(percentage, (x, y))
CPhist.yaxis.set_major_locator(ticker.LinearLocator(11))
CPhist.text(0.65, 0.95, textstr, transform=CPhist.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='left')
plt.show()

#hist of fbs
plt.figure(figsize=(5,5))
fbshist=sns.countplot(x=data['fbs'], data=data)
plt.title('fbs prior probabilities')
plt.xlabel('Category')
plt.ylabel('Count')
plt.ylim(0,220)
textstr = '\n'.join((
   r'0 = below or equal to 120 mg/dl',
   r'1 = else'))
for p in fbshist.patches:
      percentage = '{:.1f}%'.format(100 * p.get_height()/total)
      x = p.get_x() + p.get_width()/2
      y = p.get_y() + p.get_height() -0.02
      fbshist.annotate(percentage, (x, y))
fbshist.yaxis.set_major_locator(ticker.LinearLocator(11))
fbshist.text(0.50, 0.95, textstr, transform=fbshist.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='left')
plt.show()

# hist of restecg
plt.figure(figsize=(7,7))
restecghist=sns.countplot(x=data['restecg'], data=data)
plt.title('restecg prior probabilities')
plt.xlabel('Category')
plt.ylabel('Count')
plt.ylim(0,150)
textstr = '\n'.join((
   r'0 = normal',
   r'1 = having ST-T wave abnormality',
   r"2 = showing probable or definite left ventricular hypertrophy by Este's criteria"))
for p in restecghist.patches:
      percentage = '{:.1f}%'.format(100 * p.get_height()/total)
      x = p.get_x() + p.get_width()/2
      y = p.get_y() + p.get_height() -0.02
      restecghist.annotate(percentage, (x, y))
restecghist.yaxis.set_major_locator(ticker.LinearLocator(11))
restecghist.text(0.10, 0.95, textstr, transform=restecghist.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='left')
plt.show()

# hist of exang
plt.figure(figsize=(5,5))
exanghist=sns.countplot(x=data['exang'], data=data)
plt.title('exang prior probabilities')
plt.xlabel('Category')
plt.ylabel('Count')
plt.ylim(0,160)
textstr = '\n'.join((
   r'0 = no',
   r'1 = yes'))
for p in exanghist.patches:
      percentage = '{:.1f}%'.format(100 * p.get_height()/total)
      x = p.get_x() + p.get_width()/2
      y = p.get_y() + p.get_height() -0.02
      exanghist.annotate(percentage, (x, y))
exanghist.yaxis.set_major_locator(ticker.LinearLocator(11))
exanghist.text(0.65, 0.95, textstr, transform=exanghist.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='left')
plt.show()

#hist of slope
plt.figure(figsize=(5,5))
slopehist=sns.countplot(x=data['slope'], data=data)
plt.title('slope prior probabilities')
plt.xlabel('Category')
plt.ylabel('Count')
plt.ylim(0,140)
textstr = '\n'.join((
   r'0 = upsloping',
   r'1 = flat',
   r'2 = downsloping'))
for p in slopehist.patches:
      percentage = '{:.1f}%'.format(100 * p.get_height()/total)
      x = p.get_x() + p.get_width()/2
      y = p.get_y() + p.get_height() -0.02
      slopehist.annotate(percentage, (x, y))
slopehist.yaxis.set_major_locator(ticker.LinearLocator(11))
slopehist.text(0.65, 0.95, textstr, transform=slopehist.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='left')
plt.show()

#hist of ca
plt.figure(figsize=(5,5))
cahist=sns.countplot(x=data['ca'], data=data)
plt.title('ca prior probabilities')
plt.xlabel('Category')
plt.ylabel('Count')
plt.ylim(0,150)
textstr = '\n'.join((
   r'0 = 0 major vessels',
   r'1 = 1 major vessel',
   r'2 = 2 major vessels',
   r'3 = 3 major vessels',
   r'4 = unknown'))
for p in cahist.patches:
      percentage = '{:.1f}%'.format(100 * p.get_height()/total)
      x = p.get_x() + p.get_width()/2
      y = p.get_y() + p.get_height() -0.02
      cahist.annotate(percentage, (x, y))
cahist.yaxis.set_major_locator(ticker.LinearLocator(11))
cahist.text(0.65, 0.95, textstr, transform=cahist.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='left')
plt.show()

#hist of thal
plt.figure(figsize=(5,5))
thalhist=sns.countplot(x=data['thal'], data=data)
plt.title('thal prior probabilities')
plt.xlabel('Category')
plt.ylabel('Count')
plt.ylim(0,140)
textstr = '\n'.join((
   r'0 = missing value',
   r'1 = fixed defect',
   r'2 = normal',
   r'3 = reversable defect'))
for p in thalhist.patches:
      percentage = '{:.1f}%'.format(100 * p.get_height()/total)
      x = p.get_x() + p.get_width()/2
      y = p.get_y() + p.get_height() -0.02
      thalhist.annotate(percentage, (x, y))
thalhist.yaxis.set_major_locator(ticker.LinearLocator(11))
thalhist.text(0.10, 0.95, textstr, transform=thalhist.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='left')
plt.show()

#hist of age
plt.hist(data['age'], bins=10)
plt.title('age histogram')
plt.xlabel('age (years)')
plt.ylabel('frequency')
plt.xlim(20,100)
plt.show()

#hist of trestbps
plt.hist(data['trestbps'], bins=30)
plt.title('trestbps histogram')
plt.xlabel('resting blood pressure (mm/Hg)')
plt.ylabel('frequency')
plt.show()

# hist of chol
plt.hist(data['chol'], bins=40)
plt.title('chol histogram')
plt.xlabel('serum cholesterol (mg/dl)')
plt.ylabel('frequency')
plt.show()

#hist of thalach
plt.hist(data['thalach'], bins=30)
plt.title('thalach histogram')
plt.xlabel('maximum heart rate achieved')
plt.ylabel('frequency')
plt.show()

#hist of oldpeak
plt.hist(data['oldpeak'], bins=40)
plt.title('oldpeak histogram')
plt.xlabel('ST depression induced by exercise relative to rest')
plt.ylabel('frequency')
plt.show()


##--------Heatmap of correlation between continuous variables-------------------##

dataCV=data.drop(['id','gender' , 'cp' , 'chol' , 'fbs' , 'restecg', 'exang' , 'slope' ,'ca' , 'thal' ], axis=1)
sns.heatmap(dataCV.corr(), annot=True, cmap='coolwarm') #heatmap dataCV
plt.show()


##--------Connections graph between continuous variables------------------------------##

# thalach and trestbps
plt.scatter(x=data['thalach'], y=data['trestbps'])
plt.xlabel('thalach')
plt.ylabel('trestbps')
plt.show()

# thalach and age
plt.scatter(x=data['thalach'], y=data['age'])
plt.xlabel('thalach')
plt.ylabel('age')
plt.show()

##--------Correlation of correlation between categorical variables------------------##

#gender vs categorical variables
print ('\n' + "gender vs categorical variables : " + '\n' )

table = pd.crosstab(data.gender, data.cp)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.gender, data.fbs)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.gender, data.exang)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

#cp vs categorical variables
print ('\n' + "cp vs categorical variables : " + '\n' )

table = pd.crosstab(data.cp, data.exang)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.cp, data.fbs)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.cp, data.restecg)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.cp, data.slope)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.cp, data.ca)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.cp, data.thal)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

#fbs vs categorical variables
print ('\n' + "fbs vs categorical variables : " + '\n' )

table = pd.crosstab(data.fbs, data.restecg)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.fbs, data.exang)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.fbs, data.slope)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.fbs, data.ca)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

#restecg vs categorical variables
print ('\n' + "restecg vs categorical variables : " + '\n' )

table = pd.crosstab(data.restecg, data.exang)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.restecg, data.slope)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.restecg, data.ca)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

#exang vs categorical variables
print ('\n' + "exang vs categorical variables : " + '\n' )

table = pd.crosstab(data.slope, data.exang)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.restecg, data.exang)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.thal, data.exang)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.ca, data.exang)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

#slope vs categorical variables
print ('\n' + "slope vs categorical variables : " + '\n' )

table = pd.crosstab(data.slope, data.ca)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

table = pd.crosstab(data.slope, data.thal)
chi2, p, dof, expected = chi2_contingency(table.values)
print('chi2 = ' , chi2 , ' p = ' , p , ' dof = ' , dof ,' expected=' , expected )

##--------Connections graph between categorical variables------------------------------##

# exang and cp
sns.stripplot(x='exang', y='cp', data=data, jitter=0.2)
plt.show()

# exang and slope
mosaic(data,['slope', 'exang'] , axes_label=True ,title = "Categorial slope and exang : " )
plt.xlabel('slope')
plt.show()

###-------------------Changing missing values -------------------------------###

# CA
data.loc[data['ca'] == 4,'ca'] = 0

#Thal
data.loc[data['thal'] == 0,'thal'] = 2

##--------Connections graph between y and variables------------------------------##

# y and exang
mosaic(data,['exang', 'y'], gap=0.01, axes_label=True)
plt.title("heart attack (Y) and exercise induced angina (exang)", fontsize=25)
plt.show()

# y and thal
mosaic(data,['thal', 'y'], gap=0.01, axes_label=True)
plt.title("heart attack (Y) and heart defect level (thal)", fontsize=25)
plt.show()

# y and CA
sns.stripplot(x='y', y='ca', data=data, jitter=0.2)
plt.show()

# y and oldpeak
sns.boxplot(x = 'y', y='oldpeak', data=data)
plt.show()

###-------------------Converting to categorical variable -------------------------------###

# graph before changing
sns.stripplot(x='y', y='chol', data=data, jitter=0.2)
plt.show()

# Chol changing to categorical variable
data.loc[data['chol'] <= 200, 'chol'] = 0  #change chol to 0\1 , 0= normal 1=high
data.loc[data['chol'] > 300, 'chol'] = 2  #change chol to 0\1 , 0= normal 1=high
data.loc[data['chol'] > 200,'chol'] = 1  #change chol to 0\1 , 0= normal 1=high

# graph after changing
mosaic(data,['y', 'chol'] , axes_label=True ,title = "Categorial y and chol : " )
plt.xlabel('y')
plt.show()

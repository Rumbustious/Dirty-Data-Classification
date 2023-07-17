
import numpy as np 
import pandas as pd
import matplotlib.pylab as plt 

from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('./loan_data_set.csv')
print(df.describe())
print(df.head())
print(df.info())
print(df.duplicated())

# Categorical columns
cat_col = [col for col in df.columns if df[col].dtype == 'object']
print('Categorical columns :',cat_col)
# Numerical columns
num_col = [col for col in df.columns if df[col].dtype != 'object']
print('Numerical columns :',num_col)


#Check the total number of unique values in the Categorical columns
print(df[cat_col].nunique())




print(df.shape)
print(round((df.isnull().sum()/df.shape[0])*100,2))



df.dropna(subset=['Gender', 'Married', "Dependents", 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'], axis=0, inplace=True)
print(df.isnull().sum())
print(df.shape)


# print(df['ApplicantIncome'].unique()[:500])

plt.boxplot(df['LoanAmount'], vert=False)
plt.ylabel('Variable')
plt.xlabel('Income')
plt.title('Box Plot')
plt.show()



 


# Drop the outliers
for col in num_col:
    # if col == 'Loan_Amount_Term': break
    mean = df[col].mean()
    std  = df[col].std()
 
# Calculate the lower and upper bounds
    lower_bound = mean - std*2
    upper_bound = mean + std*2
    df = df[(df[col] >= lower_bound)
                        & (df[col] <= upper_bound)]
    plt.boxplot(df[col], vert=False)
    plt.ylabel('Variable')
    plt.xlabel(col)
    plt.title('Box Plot')
    plt.show()






df = df.apply(LabelEncoder().fit_transform)
# def data2vector(data):
#     names = data.columns[:]
#     for i in names:
#         col = pd.Categorical(data[i])
#         data[i] = col.codes
#     return data
# df = data2vector(df)


print(df.head())
X = df.iloc[:,:-1]
y = df.iloc[:, -1].values
print(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
mnb = MultinomialNB(alpha=1.9)         # alpha by default is 1. alpha must always be > 0. 

mnb.fit(X_train,y_train)
y_pred1 = mnb.predict(X_test)
print("Accuracy Score for Naive Bayes : ", accuracy_score(y_pred1,y_test))

rfc = RandomForestClassifier(n_estimators=100,criterion='gini')
# n_estimators = No. of trees in the forest
# criterion = basis of making the decision tree split, either on gini impurity('gini'), or on infromation gain('entropy')
rfc.fit(X_train,y_train)
y_pred3 = rfc.predict(X_test)
print("Accuracy Score of Random Forest Classifier : ", accuracy_score(y_pred3,y_test))


svc = SVC(C=1.0,kernel='rbf',gamma='auto')         
# C here is the regularization parameter. Here, L2 penalty is used(default). It is the inverse of the strength of regularization.
# As C increases, model overfits.
# Kernel here is the radial basis function kernel.
# gamma (only used for rbf kernel) : As gamma increases, model overfits.
svc.fit(X_train,y_train)
y_pred2 = svc.predict(X_test)
print("Accuracy Score for SVC : ", accuracy_score(y_pred2,y_test))


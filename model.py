import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score, roc_curve

import joblib


data = pd.read_csv("Teleco.csv")


print(data.head())


print(data.isnull().sum())


print(data.info)

#Dropping the customer ID column as it wont be relevant
data.drop('customerID', axis=1,inplace=True)

#Converting Total Charges to numeric and handling missing data
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors='coerce')
#Here coerce convert value to Nan if those data are not in numeric type like 'N/A','?' or others
data['TotalCharges'].fillna(data['TotalCharges'].mean(),inplace =True)
#We can also see if there is any other categorical data in our file:
categorical_columns = data.select_dtypes(include=['object','category']).columns
print("Categorical Columns:")
print(categorical_columns)


#Encoding categorical variables
le = LabelEncoder()
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

#Now, Let's scale numerical features
scaler = StandardScaler()
numerical_features = ['tenure','MonthlyCharges','TotalCharges']
data[numerical_features] = scaler.fit_transform(data[numerical_features])


#Defining Features (X) and target(Y)
X = data.drop('Churn',axis=1)
y = data['Churn']

#Splitting the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=12)
#Here random_state acts as a seed for the random number generator that determines how the data is shuffled and split.

#Initializing and training the model
logreg = LogisticRegression(max_iter=500, penalty='l2',solver='liblinear')
#Here max_iter specifies the number of iteration the optimization algo is allowed to run
# and alos involves an optimization process (Eg. Gradient Descent) to minimize the loss function.

#penalty specifies the regularization which solves overfitting and underfitting probles.
#L2: Ridge regularization, it adds penalty proportion to the square of the magnitude of coffiencents
#It helos to keep the model simpler by shrinking less important coeffcients towards zero.

#solver = 'liblinear'
#it specifies the algo used to solve the optimization problem.
#liblinear --> it is a library designed for small to medium sized datasets
#It works well with both l1 and l2 regularization
#alternative solutons includes saga & lbfgs

logreg.fit(X_train,y_train)

#Predict on the test set
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:,1]


#Evaluation of the model
conf_matrix = confusion_matrix(y_test,y_pred)
# sns.heatmap(conf_matrix, annot=True, fmt ='d',cmap="Blues")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel('Actual')
# plt.show()


#Classificaiton report
print("Classification Report:\n", classification_report(y_test,y_pred))


#Roc curve and AUC score
fpr,tpr,_ = roc_curve(y_test,y_prob)
auc_score = roc_auc_score(y_test,y_pred)


# plt.plot(fpr,tpr,label=f"AUC = {auc_score:.2f}")
# plt.plot([0,1],[0,1],linestyle="--",color='gray')
# plt.title("ROC Curve")
# plt.xlabel("Flase Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.show()



#Interpreting model coefficients
#Reterive coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logreg.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("Feature Importance: \n", coefficients)


joblib.dump(logreg, "logistic_model.pkl")
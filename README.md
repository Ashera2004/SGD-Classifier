# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Data Preparation**  
   - Collect or load the dataset.  
   - Identify **features (X)** and **target labels (y)**.  
   - Split the dataset into **training** and **testing** subsets.  

2. **Preprocessing**  
   - Normalize or standardize the feature set to improve convergence.  
   - Handle missing values or categorical variables if present.  

3. **Initialize and Train the SGD Classifier**  
   - Define the **SGD Classifier** with an appropriate loss function (e.g., **hinge loss** for SVM, **log loss** for logistic regression).  
   - Set parameters like learning rate, maximum iterations, and regularization if needed.  
   - Train the model using the **stochastic gradient descent** optimization method.  

4. **Prediction**  
   - Use the trained model to predict labels for the test dataset.  
   - Apply the `.predict()` method on new or unseen data.  

5. **Model Evaluation**  
   - Compute accuracy, precision, recall, and F1-score.  
   - Generate a **confusion matrix** to analyze misclassifications.
  

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics  import accuracy_score, confusion_matrix,classification_report

iris=load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())

X=df.drop('target',axis=1)
Y=df['target']
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2, random_state=42)
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,Y_train)
Y_pred=sgd_clf.predict(X_test)

accuracy=accuracy_score(Y_test,Y_pred)
print(f"Accuracy: {accuracy: .3f}")

cm=confusion_matrix(Y_test,Y_pred)
print(f"Confusion Matrix:")
print(cm)

classification_report1=classification_report(Y_test,Y_pred)
print(classification_report1)

Developed by: A S Siddarth
RegisterNumber: 212224040316
*/
```

## Output:

![Screenshot 2025-03-30 210635](https://github.com/user-attachments/assets/00bed5bb-ec3b-407c-b7cb-e3f8f4c9b12d)

![Screenshot 2025-03-30 210642](https://github.com/user-attachments/assets/4b87b172-2101-4059-be79-9c430ce6a2b0)

![Screenshot 2025-03-30 210715](https://github.com/user-attachments/assets/cfbbc669-d6ee-443f-a34a-260ba23abbe3)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

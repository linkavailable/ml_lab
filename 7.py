import pandas as pd

msg=pd.read_csv('naive.csv',names=['message','label']) 
print('Total instances in the dataset:',msg.shape[0])
msg['labelnum']=msg.label.map({'Yes':1,'No':0})
X=msg.message
Y=msg.labelnum

from sklearn.model_selection import train_test_split 
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.25) 
print('\nDataset is split into Training and Testing samples') 
print('Total training instances :', xtrain.shape[0]) 
print('Total testing instances :', xtest.shape[0])
from sklearn.feature_extraction.text import CountVectorizer 
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain) 
xtest_dtm = count_vect.transform(xtest)
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(xtrain_dtm,ytrain)
predicted=clf.predict(xtest_dtm)
from sklearn import metrics
print("\nAccuracy metrics")
print("\nAccuracy of the classifier is:",metrics.accuracy_score(ytest,predicted))
print("Recall:",metrics.recall_score(ytest,predicted))
print("Confusion matrix",metrics.confusion_matrix(ytest,predicted))
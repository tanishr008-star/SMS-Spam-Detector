import pandas as pd

messages=pd.read_csv('C:/SpyderPython/sms+spam+collection/sms+spam+collection/SMSSpamCollection',sep='\t',names=['label','message'])


import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-z A-Z]', ' ' , messages['message'][i])
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word  in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

#Creating a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
x= cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


#training model using naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detection_model=MultinomialNB().fit(x_train,y_train)
#prediction
y_predict=spam_detection_model.predict(x_test)


#testing results
from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_predict)

#accuracy score
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)

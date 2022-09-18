# Reentika Awasthi
# Language Detection Model

# import all  libraries
import pandas as pd # helps with data analyisis 
import numpy as np  # mathematical functions
import re # characters that form a speech pattern 
import seaborn as sns # data visulaization 
import matplotlib.pyplot as plt # data visulaization
import warnings
warnings.simplefilter("ignore")


# read/load the dataset
data = pd.read_csv("Language Detection.csv")

# counting  values of all languages
value_count = data["Language"].value_counts()

# seperate independent/x + dependent/y variables
independent = data["Text"]
dependent = data["Language"]

# converting categorical variables to numerical
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dependent = label_encoder.fit_transform(dependent)

# creating a list for appending the preprocessed text
data_list = []

# go through all text
for text in independent:
    # remove all symbols and numbers (if any)
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    # convert all text to lowercase 
    text = text.lower()
    # appending to the list created above
    data_list.append(text)

#  a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
independent = cv.fit_transform(data_list).toarray()
independent.shape 

# split train and testing dataset
from sklearn.model_selection import train_test_split
i_train, i_test,  d_train, d_test = train_test_split(independent, dependent, test_size=0.2)

# training the model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(i_train, d_train)

# predict output of test set
d_pred = model.predict(i_test)

# evalute the accuracy of our model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
ac = accuracy_score(d_test, d_pred)
cm = confusion_matrix(d_test, d_pred)

#print("Accuracy is :",ac)

# plot the confusion matrix
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()

# predict the data

def predict(text):
     independent = cv.transform([text]).toarray() # convert text to bag of words model
     lang = model.predict(independent) # predict the language
     lang = label_encoder.inverse_transform(lang) # find the language that relates to the predicted value
     print("The langauge is in", lang[0]) # print the language!

while text != "stop":
    text = input("Enter word(s): ").lower()
    predict(text)

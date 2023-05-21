import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import email
import string
from bs4 import BeautifulSoup
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve
import os
import warnings
warnings.filterwarnings('ignore')
np.random.seed(49)


# loading the data
easy_ham_path = '/kaggle/input/spamassassin-public-corpus/easy_ham/easy_ham/'
hard_ham_path = '/kaggle/input/spamassassin-public-corpus/hard_ham/hard_ham/'
spam_path = '/kaggle/input/spamassassin-public-corpus/spam_2/spam_2/'


def get_data(path):
    data = []
    files = os.listdir(path)
    for file in files:
        f = open(path+file, encoding = "ISO-8859-1")
        words_list = f.read()
        data.append(words_list)
        f.close()
    return data

easy_ham = get_data(easy_ham_path)
hard_ham = get_data(hard_ham_path)
ham = easy_ham + hard_ham
spam = get_data(spam_path)

np.random.shuffle(ham)
np.random.shuffle(spam)

print(spam[49])

# data preprocessing 
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
class email_to_clean_text(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        text_list = []
        for mail in X:
            b = email.message_from_string(mail)
            body = ""

            if b.is_multipart():
                for part in b.walk():
                    ctype = part.get_content_type()
                    cdispo = str(part.get('Content-Disposition'))

                    # skip any text/plain (txt) attachments
                    if ctype == 'text/plain' and 'attachment' not in cdispo:
                        body = part.get_payload(decode=True)  # get body of email
                        break
            # not multipart - i.e. plain text, no attachments, keeping fingers crossed
            else:
                body = b.get_payload(decode=True) # get body of email
            #####################################################
            soup = BeautifulSoup(body, "html.parser") #get text from body (HTML/text)
            text = soup.get_text().lower()
            #####################################################
            text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE) #remove links
            ####################################################
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text, flags=re.MULTILINE) #remove email addresses
            ####################################################
            text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
            ####################################################
            text = ''.join([i for i in text if not i.isdigit()]) # remove digits
            ####################################################
            stop_words = stopwords.words('english')
            words_list = [w for w in text.split() if w not in stop_words] # remove stop words
            ####################################################
            words_list = [lemmatizer.lemmatize(w) for w in words_list] #lemmatization
            ####################################################
            words_list = [stemmer.stem(w) for w in words_list] #Stemming
            text_list.append(' '.join(words_list))
        return text_list
    
email_to_text = email_to_clean_text()
text_ham = email_to_text.transform(ham)
text_spam = email_to_text.transform(spam)


# Data Visualization
text_easy_ham = email_to_text.transform(easy_ham)
text_hard_ham = email_to_text.transform(hard_ham)
data = [len(ham)/len(ham+spam), len(spam)/len(ham+spam)]
labels = ['ham', 'spam']
colors = ['green', 'red']
plt.figure(figsize=(12, 5))
plt.pie(data, labels = labels, autopct='%.0f%%', colors=colors)
plt.show()
plt.figure(figsize=(8, 5))
sns.countplot(x = ['ham']*len(ham) + ['spam']*len(spam), palette=colors)
plt.show()

def plot_WordCloud(text_list):
    unique_string=(" ").join(text_list)
    wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

plot_WordCloud(text_easy_ham)
plot_WordCloud(text_hard_ham)
plot_WordCloud(text_spam)

# spilitting the dataset
y = len(text_ham)*[0] + len(text_spam)*[1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_ham+text_spam, y,
                                                    stratify=y, 
                                                    test_size=0.2)


# vectorization
vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train).toarray()
y_train = np.array(y_train).reshape(len(y_train), 1)
X_test = vectorizer.transform(X_test).toarray()
y_test = np.array(y_test).reshape(len(y_test), 1)

# model building and training
import numpy as np
import os
import re
import string
import email
import warnings
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')
np.random.seed(49)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

class email_to_clean_text(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None): 
        return self
    def transform(self, X, y=None):
        stop_words = stopwords.words('english')
        text_list = []
        for mail in X:
            text = BeautifulSoup(email.message_from_string(mail).get_payload(decode=True), "html.parser").get_text().lower() if email.message_from_string(mail).get_payload(decode=True) is not None else ""
            text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text, flags=re.MULTILINE)
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = ''.join([i for i in text if not i.isdigit()])
            words_list = [stemmer.stem(lemmatizer.lemmatize(w)) for w in text.split() if w not in stop_words]
            text_list.append(' '.join(words_list))
        return text_list

paths = ['/Users/drewjordan/Documents/AI-spam-detector/Spam-Email-Classifier/Data/archive/easy_ham/',
         '/Users/drewjordan/Documents/AI-spam-detector/Spam-Email-Classifier/Data/archive/hard_ham/',
         '/Users/drewjordan/Documents/AI-spam-detector/Spam-Email-Classifier/Data/archive/spam/']


data = []
labels = []
for i, path in enumerate(paths):
    data += [open(path+file, encoding = "ISO-8859-1").read() for file in os.listdir(path)]
    labels += [i]*len(os.listdir(path))

np.random.seed(42)
combined = list(zip(data, labels))
np.random.shuffle(combined)
data, labels = zip(*combined)

text_transformer = email_to_clean_text()
text = text_transformer.transform(data)

X_train, X_test, y_train, y_test = train_test_split(text, labels, stratify=labels, test_size=0.2)

vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

rfc = RandomForestClassifier(n_estimators=1200, oob_score=True)
rfc.fit(X_train, y_train)

predictions = rfc.predict(X_test)

print("accuracy score = {}%".format(round(accuracy_score(y_test, predictions)*100, 2)))
print("f1 score = {}".format(round(f1_score(y_test, predictions, average='weighted'), 2)))
conf_mx = confusion_matrix(y_test, predictions)

y_scores = cross_val_predict(rfc, X = X_train, y = y_train, cv=5, method="predict_proba")
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores[:,1])
fpr, tpr, thresholds = roc_curve(y_train, y_scores[:,1])
roc_auc_score(y_train, y_scores[:,1])

# Replacing the old pipeline with the revised one
my_pipeline = Pipeline(steps=[
    ('text', email_to_clean_text()),
    ('vector', CountVectorizer(stop_words='english')),
    ('model', RandomForestClassifier(n_estimators=1200, oob_score=True))])

my_pipeline.fit(data, labels)

my_pipeline.predict([data[77]])

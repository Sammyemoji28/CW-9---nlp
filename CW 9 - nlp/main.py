
# - nlp : natural language processing

import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# - loading train and test data
trainData = pd.read_csv("train.txt", delimiter = ";", names = ["text","label"])
testData = pd.read_csv("test.txt", delimiter = ";", names = ["text","label"])
# print(trainData.head())
# print(testData.head())

# - checking how many emotion options there are
# print(trainData["label"].value_counts())


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

# Reading the data
df = pd.read_csv('ThoracicSurgery.csv')

df.describe()

X2 = df[['Performance', 'Dyspnoea', 'Cough', 'Tumor_Size', 'Diabetes_Mellitus']]
y = df['Death_1yr']

# Developing a Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=.3, random_state=1111, stratify=y)  
clf = RandomForestClassifier(n_estimators = 5, max_depth = 7, min_samples_leaf = 8, random_state=1111) 
clf.fit(X_train, y_train)

# Saving model to the disk
pickle.dump(clf, open('model.pki','wb'))

model = pickle.load(open('model.pki','rb'))
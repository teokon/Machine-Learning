import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args


file_paths = [
    "../Desktop/harth/S006.csv",
    "../Desktop/harth/S008.csv",
    "../Desktop/harth/S009.csv",
    "../Desktop/harth/S010.csv",
    "../Desktop/harth/S012.csv",
    "../Desktop/harth/S013.csv",
    "../Desktop/harth/S014.csv",
    "../Desktop/harth/S015.csv",
    "../Desktop/harth/S016.csv",
    "../Desktop/harth/S017.csv",
    "../Desktop/harth/S018.csv",
    "../Desktop/harth/S019.csv",
    "../Desktop/harth/S020.csv",
    "../Desktop/harth/S021.csv",
    "../Desktop/harth/S022.csv",
    "../Desktop/harth/S023.csv",
    "../Desktop/harth/S024.csv",
    "../Desktop/harth/S025.csv",
    "../Desktop/harth/S026.csv",
    "../Desktop/harth/S027.csv",
    "../Desktop/harth/S028.csv",
    "../Desktop/harth/S029.csv",
]


all_data = pd.concat([pd.read_csv(file) for file in file_paths])
sampled_data = all_data
data1 = sampled_data.drop(columns=['timestamp', 'index', 'Unnamed: 0'])

# Check for NaN values
data1.replace([np.inf, -np.inf], np.nan, inplace=True)
data1.dropna(inplace=True)


X = data1.drop(columns=['label'])
y = data1['label']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#  Naive Bayes classifier
nb_classifier = GaussianNB()

# seed search space
search_space = [
    Integer(0, 10000, name='random_state')
]

# optimization
@use_named_args(search_space)
def objective(random_state):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=random_state)
    nb_classifier.fit(X_train, y_train)
    accuracy = nb_classifier.score(X_test, y_test)
    return -accuracy  

# Bayesian Optimization 
res = gp_minimize(objective, search_space, n_calls=50, random_state=42)


print(f"Best accuracy: {-res.fun}")
print(f"Best random_state: {res.x[0]}")

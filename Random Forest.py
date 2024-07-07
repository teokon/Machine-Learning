import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics

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
    "../Desktop/harth/S029.csv"
]


all_data = pd.concat([pd.read_csv(file) for file in file_paths])
sampled_data = all_data.sample(frac=0.3)
sampled_data = sampled_data.drop(columns=['timestamp', 'index', 'Unnamed: 0'])
labels = sampled_data[['label']]  

X = sampled_data.drop('label', axis=1) 
y = labels


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

# RandomForestClassifier με παράλληλη εκπαίδευση
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# classification report
target_names = list(map(str, labels['label'].unique()))
report = classification_report(y_test, y_pred, target_names=target_names)
print("\nClassification Report:\n")
print(report)

# accuracy
print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))

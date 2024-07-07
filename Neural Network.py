import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report


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
sampled_data = all_data.sample(frac=0.1)
sampled_data = sampled_data.drop(columns=['timestamp', 'index', 'Unnamed: 0'])

# One-hot encode the labels
encoder = OneHotEncoder()
labels = encoder.fit_transform(sampled_data[['label']]) 
labels = labels.toarray()

# Separate features and labels
X = normalize(sampled_data.drop('label', axis=1))  
y = labels

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(12, activation='softmax')) 


print(model.summary())


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop])

preds = (model.predict(X_test) > 0.5).astype("int32")

loss, accuracy = model.evaluate(preds, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

print(classification_report(y_test, preds))
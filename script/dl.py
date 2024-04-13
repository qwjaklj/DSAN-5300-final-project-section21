import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load data
file_path = '../data/data_cleaned.csv'
data = pd.read_csv(file_path)

# Fill missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

features = ['SPIN14','SPIN15','SPIN13','SPIN6', 'Age', 'Hours','SWL1', 'SWL3', 'SWL4']
X = data[features]
y = data['GAD_T']

# Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert target variable to one-hot encoding
ohe = OneHotEncoder()
y_train_ohe = ohe.fit_transform(y_train.values.reshape(-1, 1)).toarray()
y_test_ohe = ohe.transform(y_test.values.reshape(-1, 1)).toarray()

# Deep learning model
model = Sequential()
model.add(Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train_ohe.shape[1], activation='softmax'))  # Output layer with one neuron per class

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train_scaled, y_train_ohe, validation_split=0.1, epochs=50, batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(X_test_scaled, y_test_ohe)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

model.save('../dl_result/my_deep_learning_model.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('../dl_result/model_accuracy.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('../dl_result/model_loss.png')
plt.show()

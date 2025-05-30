import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('StudentsPerformance.csv')

# Membuat label kelulusan berdasarkan rata-rata nilai
# Asumsi: Lulus jika rata-rata nilai >= 60, Tidak Lulus jika < 60
data['average_score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3
data['graduation_status'] = np.where(data['average_score'] >= 60, 'Lulus', 'Tidak Lulus')

# Preprocessing data
# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
data['graduation_status_encoded'] = le_target.fit_transform(data['graduation_status'])

# Memilih fitur dan target
features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 
            'test preparation course', 'math score', 'reading score', 'writing score']
X = data[features]
y = data['graduation_status_encoded']

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membangun model ANN
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Kompilasi model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Melatih model
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=1, 
                    batch_size=32,
                    verbose=1)

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nAkurasi model: {accuracy*100:.2f}%")

# Prediksi
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualisasi akurasi dan loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Contoh prediksi
sample_data = X_test[:5]
predictions = model.predict(sample_data)
predicted_classes = (predictions > 0.5).astype(int)

print("\nContoh Prediksi:")
for i in range(5):
    print(f"Data {i+1}: Prediksi = {le_target.inverse_transform(predicted_classes[i:i+1])[0]}, Aktual = {le_target.inverse_transform([y_test.iloc[i]])[0]}")
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Path to your dataset folder
dataset_path = 'dataset'

# Function to extract MFCC features
def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Initialize lists
features = []
labels = []

# Loop through the dataset folder and extract features
for label in ['on', 'off']:  # Make sure your folder names are exactly 'on' and 'off'
    label_folder = os.path.join(dataset_path, label)
    for file in os.listdir(label_folder):
        if file.endswith('.wav'):
            audio_path = os.path.join(label_folder, file)
            try:
                mfcc_features = extract_mfcc(audio_path)
                features.append(mfcc_features)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Show the shape
print(f"Features Shape: {features.shape}")
print(f"Labels Shape: {labels.shape}")

# Visualize distribution
on_count = np.sum(labels == 'on')
off_count = np.sum(labels == 'off')

plt.figure(figsize=(6, 4))
plt.bar(['on', 'off'], [on_count, off_count], color=['blue', 'orange'], edgecolor='black')
plt.title("Distribution of 'on' and 'off' Commands")
plt.xlabel("Commands")
plt.ylabel("Frequency")
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Train model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predict
predictions = classifier.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions) * 100
report = classification_report(y_test, predictions, target_names=['on', 'off'], output_dict=True)

on_precision = report['on']['precision'] * 100
off_precision = report['off']['precision'] * 100
on_recall = report['on']['recall'] * 100
off_recall = report['off']['recall'] * 100
on_f1 = report['on']['f1-score'] * 100
off_f1 = report['off']['f1-score'] * 100

# Print metrics
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision for 'on': {on_precision:.2f}%")
print(f"Precision for 'off': {off_precision:.2f}%")
print(f"Recall for 'on': {on_recall:.2f}%")
print(f"Recall for 'off': {off_recall:.2f}%")
print(f"F1-Score for 'on': {on_f1:.2f}%")
print(f"F1-Score for 'off': {off_f1:.2f}%")

# Save model
joblib.dump(classifier, 'speech_command_model.pkl')
print("Model saved to 'speech_command_model.pkl'")

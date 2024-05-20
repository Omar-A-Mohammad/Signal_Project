import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import wfdb
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings("ignore")

# Importing Data
os.chdir("/kaggle/input/mit-bih")

# Extracting file names without extension
directory = os.listdir()
directory = [x.split('.')[0] for x in directory]
directory.remove('102-0')
directory.remove('RECORDS')
directory.remove('SHA256SUMS')
directory.remove('ANNOTATORS')
directory = list(set(directory))

# Data Visualization
record = wfdb.rdrecord('100', sampto=1000, channels=[0])
ann = wfdb.rdann('100', 'atr', sampto=1000)
wfdb.plot_wfdb(record, annotation=ann, figsize=(20, 5))
print(ann.__dict__)

record = wfdb.rdrecord('100', channels=[0])
print(len(record.__dict__['p_signal']))

# Data Preprocessing
def extract_beats_and_labels(sig_directory):
    window_size = 256
    Full_Signal = wfdb.rdrecord(sig_directory, sampto=650000, channels=[0]).__dict__['p_signal'].flatten()
    ann = wfdb.rdann(sig_directory, 'atr', sampto=650000)
    ann_pos = ann.__dict__['sample'][1:-1]
    ann_sym = ann.__dict__['symbol'][1:-1]

    data_Full = []
    data_Sym = []
    for QRS_pos, Beat_diagnose in zip(ann_pos, ann_sym):
        start = QRS_pos - window_size // 2
        end = QRS_pos + window_size // 2
        signal_corpus = Full_Signal[start:end]
        if len(signal_corpus) == 256:
            data_Full.append(list(signal_corpus))
            data_Sym.append(Beat_diagnose)
    return data_Full, data_Sym

# Extracting beats and labels for all signals in the directory
X = []
Y = []
for sig_name in directory:
    Full_Signal, annotation_symbol = extract_beats_and_labels(sig_name)
    X.extend(Full_Signal)
    Y.extend(annotation_symbol)
X = np.array(X)
Y = np.array(Y)

print(X)
print(Y)
print(set(Y))

Normal_mask = Y == 'N'
abnormal_mask = Y != 'N'
Normal_data = X[Normal_mask]
abnormal_data = X[abnormal_mask]

df_n = pd.DataFrame(Normal_data)
df_n["L"] = "N"
df_abn = pd.DataFrame(abnormal_data)
df_abn["L"] = "ABN"

df = pd.concat([df_n, df_abn], ignore_index=True)
print(df)
print(df["L"].value_counts())
df["L"].value_counts().plot(kind="bar")

label_dictionary = {"N": 0, "ABN": 1}
df["L"] = df["L"].map(label_dictionary)

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

sampling_strategy = {0: 37000, 1: 37000}
rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=30)
X_resampled, y_resampled = rus.fit_resample(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.8, random_state=30)

print(X_resampled.shape)
print(y_resampled.shape)
print("*" * 20)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Evaluation Metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = (model.predict(X_train) > 0.5).astype("int32")
    y_test_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Confusion matrix for training set
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    
    # Confusion matrix for testing set
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    
    # Plotting the confusion matrix for training set
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Training Set')
    plt.show()
    
    # Plotting the confusion matrix for testing set
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Testing Set')
    plt.show()
    
    # Classification report for training set
    print('Classification Report - Training Set')
    print(classification_report(y_train, y_train_pred))
    
    # Classification report for testing set
    print('Classification Report - Testing Set')
    print(classification_report(y_test, y_test_pred))

# Function to plot loss curves and ROC curves
def plot_curves(history, model, X_train, y_train, X_test, y_test):
    # Plotting the training and testing loss curves
    plt.figure(figsize=(10, 7))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.show()

    # ROC curve for training set
    fpr_train, tpr_train, _ = roc_curve(y_train, model.predict(X_train))
    roc_auc_train = auc(fpr_train, tpr_train)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_train)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Training Set')
    plt.legend(loc="lower right")
    plt.show()

    # ROC curve for testing set
    fpr_test, tpr_test, _ = roc_curve(y_test, model.predict(X_test))
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_test)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Testing Set')
    plt.legend(loc="lower right")
    plt.show()

# Training Model - DNN (Deep Neural Network)
DNN_model = Sequential()
DNN_model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
DNN_model.add(Dropout(rate=0.25))
DNN_model.add(Dense(1, activation='sigmoid'))
DNN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

DNN_model.summary()

history = DNN_model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

evaluate_model(DNN_model, X_train, y_train, X_test, y_test)
plot_curves(history, DNN_model, X_train, y_train, X_test, y_test)

# Training Model - CNN
CNN_model = Sequential()
CNN_model.add(Conv1D(256, 7, activation='relu', input_shape=(256, 1), padding='same'))
CNN_model.add(MaxPooling1D(5))
CNN_model.add(Dropout(0.2))
CNN_model.add(Conv1D(128, 5, padding='same', activation='relu'))
CNN_model.add(MaxPooling1D(5))
CNN_model.add(Conv1D(64, 5, padding='same', activation='relu'))
CNN_model.add(MaxPooling1D(5))
CNN_model.add(GlobalAveragePooling1D())
CNN_model.add(Dense(50, activation='relu'))
CNN_model.add(Dense(10, activation='relu'))
CNN_model.add(Dense(1, activation='sigmoid'))
CNN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

CNN_model.summary()

history = CNN_model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

evaluate_model(CNN_model, X_train, y_train, X_test, y_test)
plot_curves(history, CNN_model, X_train, y_train, X_test, y_test)

# Training Model - XGBoost
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
history = xgb_model.fit(X_train, y_train)

y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

evaluate_model(xgb_model, X_train, y_train, X_test, y_test)

# Training Model - Decision Tree
DT_model = DecisionTreeClassifier(random_state=30)
history = DT_model.fit(X_train, y_train)

y_train_pred = DT_model.predict(X_train)
y_test_pred = DT_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

evaluate_model(DT_model, X_train, y_train, X_test, y_test)

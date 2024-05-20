
# ECG Classification using Machine Learning and Deep Learning

## Introduction
<p>Electrocardiogram (ECG) signals are a vital tool in diagnosing various heart conditions. This project focuses on classifying ECG beats into normal and abnormal categories, which can aid in automated diagnosis and monitoring of cardiac health.</p>

## Project Summary
<p>We are utilizing ECG data collected by BIH and MIT, which provides valuable insights into the heart's electrical activity. Here is a summary of the dataset:</p>

<li>Number of records: 48</li>
<li>Sampling frequency: 360 samples per second</li>
<li>Data Distribution: The dataset consists of records from 25 male subjects between the ages of 32 and 89 and 22 female subjects aged from 23 to 89 years. Approximately 60% of the subjects were inpatients.</li>

## Dataset
<p>The MIT-BIH Arrhythmia Database is used for this project, which contains annotated ECG recordings with different beat types
  https://www.kaggle.com/datasets/mostafa1221/mit-bih.</p>

## Preprocessing
<ul>
  <li>ECG signals are extracted from the MIT-BIH database.</li>
  <li>Beats are segmented and labeled.</li>
  <li>Data is preprocessed and prepared for model training.</li>
</ul>

## Models Implemented

### Feedforward Neural Network
<p>A simple feedforward neural network with dropout regularization.</p>

### Convolutional Neural Network (CNN)
<p>A CNN model architecture for learning spatial features from ECG signals.</p>

### XGBoost
<p>Gradient boosting algorithm for classification tasks.</p>

### Decision Tree
<p>Simple decision tree classifier.</p>

## Evaluation
<ul>
  <li>Models are evaluated using confusion matrices, classification reports, accuracy scores, and ROC curves.</li>
  <li>Loss curves are plotted to visualize training and validation performance.</li>
</ul>

## Requirements
<ul>
  <li>Python 3</li>
  <li>Libraries: numpy, pandas, seaborn, matplotlib, wfdb, scikit-learn, imbalanced-learn, keras, xgboost</li>
</ul>

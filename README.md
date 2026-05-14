# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1070" height="847" alt="Screenshot 2026-05-12 144313" src="https://github.com/user-attachments/assets/01482a77-03a3-4010-9ba0-b4a7f9b0fc2f" />


## DESIGN STEPS
### STEP 1: 
Import all the necessary libraries
### STEP 2: 
Import the dataset and do data preprocessing
### STEP 3: 
Split the data into training data and testing data
### STEP 4: 
Normalize the features, convert the data into tensors and create DataLoaders
### STEP 5: 
Create a custom neural network class, define a function for training the model and also define a function to test the model
### STEP 6: 
Give a custom data to predict the output, analyze the model using confusion matrix and classification report.




## PROGRAM

### Name:KANNAN N
### Register Number: 212223230097

```python
# Mount the drive

from google.colab import drive
drive.mount('/content/drive')
```
```py
# Import all the necessary libraries

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
```
```py
# Import the dataset and check the contents inside the dataset

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/DEEP LEARNING AND ITS APPLICATION/customers.csv')
data.head()
data.columns
```
```py
# Data Preprocessing

# Drop ID column as it's not useful for classification

data = data.drop(columns=["ID"])

# Handle missing values

data.fillna({"Work_Experience": 0, "Family_Size": data["Family_Size"].median()}, inplace=True)

# Encode categorical variables

categorical_columns = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Encode target variable

label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])  # A, B, C, D -> 0, 1, 2, 3

# Split features and target

X = data.drop(columns=["Segmentation"])
y = data["Segmentation"].values
```
```py
# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```py
# Normalize features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
```py
# Convert to tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
```
```py
# Create DataLoader

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
```
```py
# Define Neural Network(Model1)

class PeopleClassifier(nn.Module):
    def __init__(self, input_size): #Changed init to __init__
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8,4) #Changed fc3 to fc4 to avoid overwriting


    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x
```
```py
# Training Loop

def train_model(model, train_loader, criterion, optimizer, epochs):

  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```
```py
# Initialize model

model = PeopleClassifier(input_size = X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs = 100)
```
```py
# Evaluation

model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())
```
```py
# Compute metrics

accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])
print("Name:KANNAN N")
print("Register No: 212223230097")
print()
print(f'Test Accuracy: {accuracy:.2f}%')
print()
print("Confusion Matrix:\n", conf_matrix)
print()
print("Classification Report:\n", class_report)
```
```py
# Plotting the graph

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```
```py
# Prediction for a sample input

sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)
    # Select the prediction for the sample (first element)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
print("Name:KANNAN N")
print("Register No: 212223230097")
print()
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')
```
### Dataset Information

<img width="1366" height="325" alt="image" src="https://github.com/user-attachments/assets/73ce44dd-c89f-4315-ac4c-4146869729d6" />


### OUTPUT

## Confusion Matrix
<img width="334" height="120" alt="image" src="https://github.com/user-attachments/assets/5c7b42fc-7d80-41d8-94db-61736afa8474" />

## Classification Report
<img width="653" height="232" alt="image" src="https://github.com/user-attachments/assets/f458ca07-00a0-4684-b9e2-0e38bacb0b22" />


### New Sample Data Prediction
<img width="1015" height="297" alt="image" src="https://github.com/user-attachments/assets/85b8c37a-27bc-491b-86cf-e78bc2e7db56" />


## RESULT
Thus, a neural network for classification model is implemented successfully.

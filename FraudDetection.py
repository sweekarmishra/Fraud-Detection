#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
df=pd.read_csv("creditcard.csv")
df.head()


# In[2]:


print(df.info())
print(df.isnull().sum())


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt

# Count of fraudulent (1) vs legitimate (0) transactions
print(df["Class"].value_counts())

# Plot the distribution
sns.countplot(x="Class", data=df)
plt.title("Class Distribution (0 = Legit, 1 = Fraud)")
plt.show()


# In[4]:


print(df.describe())


# In[5]:


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[6]:


# Select features and target variable
X = df.drop(columns=["Class"])  # Features (all columns except 'Class')
y = df["Class"]  # Target variable (fraud or not)


# In[7]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])  # Normalize 'Amount'


# In[8]:


from sklearn.model_selection import train_test_split

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check the shape of training and testing data
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# In[33]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the model
model.fit(X_train, y_train)


# In[34]:


# Predict on test set
y_pred = model.predict(X_test)


# In[35]:


from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")


# In[36]:


from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# In[37]:


from sklearn.metrics import classification_report

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[38]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(y_test[:5])  # This should print the first 5 values


# In[39]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# In[40]:


from sklearn.metrics import classification_report

print("Classification Report:\n", classification_report(y_test, y_pred))


# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[42]:


import numpy as np

feature_importances = model.feature_importances_
feature_names = X_train.columns

# Sort features by importance
indices = np.argsort(feature_importances)[::-1]

# Plot
plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.bar(range(len(feature_names)), feature_importances[indices], align="center")
plt.xticks(range(len(feature_names)), feature_names[indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.show()


# In[43]:


import joblib

# Save the model
joblib.dump(model, "fraud_detection_model.pkl")

print("Model saved successfully!")


# In[44]:


# Load the model
loaded_model = joblib.load("fraud_detection_model.pkl")

# Make predictions using the loaded model
new_predictions = loaded_model.predict(X_test)

# Check if predictions are the same
print("Predictions match:", np.array_equal(y_pred, new_predictions))


# In[ ]:





# In[ ]:





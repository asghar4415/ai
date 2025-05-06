# Part 1: Data Preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv('customer_segmentation_dataset.csv', sep=',')


# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Spending Score'].fillna(df['Spending Score'].mean(), inplace=True)
df['Male'].fillna(df['Male'].mode()[0], inplace=True)




# Encode categorical variables
label = LabelEncoder()
df['Gender'] = label.fit_transform(df['Gender'])
df['Region'] = label.fit_transform(df['Region'])
df['Customer Segment'] = label.fit_transform(df['Customer Segment'])

# Standardize numerical features
scalar = StandardScaler()
df[['Age', 'Income', 'Spending Score']] = scalar.fit_transform(df[['Age', 'Income', 'Spending Score']])

# Segregate features and target
x = df[['Customer ID', 'Age', 'Income', 'Spending Score', 'Gender', 'Region']]
y = df['Customer Segment']





# Part 2: Data Splitting and Model Training


# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Initialize models
model_lr = LogisticRegression()
model_svc = SVC()
model_dt = DecisionTreeClassifier()



# Tuned models
t_model_lr = LogisticRegression(C=0.3, solver='liblinear')
t_model_dt = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
t_model_svc = SVC(C=0.5, kernel='linear', gamma='scale')

# Train models
t_model_lr.fit(x_train, y_train)
t_model_svc.fit(x_train, y_train)
t_model_dt.fit(x_train, y_train)

# Part 3: Model Evaluation


models = {
    'Logistic Regression': t_model_lr,
    'SVC': t_model_svc,
    'Decision Tree': t_model_dt
}

for name, model in models.items():
    print(f"\nEvaluating: {name}")
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    print("Training Performance:")
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(classification_report(y_train, y_pred_train))

    print("Testing Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test))

    # K-Fold Cross Validation
    cv_scores = cross_val_score(model, x, y, cv=5)
    print(f"Average K-Fold Score (5 folds): {cv_scores.mean():.4f}")

    # kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # scores = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')


# Prediction on new data
new_data = pd.DataFrame({
    'Customer ID': [1111],
    'Age': [46],
    'Income': [61900],
    'Spending Score': [30],
    'Gender': [1],
    'Region': [1]
})

# Preprocess new data
new_data[['Age', 'Income', 'Spending Score']] = scalar.transform(new_data[['Age', 'Income', 'Spending Score']])


# Predict using trained models
for name, model in models.items():
    result = model.predict(new_data)
    print(f"Prediction by {name}: Customer belongs to Segment {result[0]}")













#spam email


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
data = pd.read_csv(url, encoding='latin-1')
data = data[['v1', 'v2']]  # Keep only label and text columns
data.columns = ['label', 'text']

# Preprocess the data
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary

# Feature extraction: Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Spam', 'Spam'], 
            yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Test with new emails
new_emails = [
    "Congratulations! You've won a $1000 gift card. Click here to claim!",  # Likely spam
    "Hi John, just checking in about our meeting tomorrow at 2pm"  # Likely not spam
]
new_emails_transformed = vectorizer.transform(new_emails)
predictions = model.predict(new_emails_transformed)
probabilities = model.predict_proba(new_emails_transformed)

print("\nNew Email Predictions:")
for email, pred, prob in zip(new_emails, predictions, probabilities):
    print(f"\nEmail: {email[:50]}...")
    print(f"Prediction: {'Spam' if pred == 1 else 'Not Spam'}")
    print(f"Confidence: {prob[pred]*100:.1f}%")
    
    
    
    
    
    
    
    
    
    


#evaluate customers



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Create synthetic customer data
np.random.seed(42)
n_customers = 1000

data = {
    'total_spent': np.random.normal(500, 200, n_customers).clip(0),
    'visit_frequency': np.random.poisson(5, n_customers),
    'age': np.random.randint(18, 70, n_customers),
    'purchase_frequency': np.random.uniform(0.1, 1.0, n_customers)
}

# Create target variable (high-value = 1, low-value = 0)
data['high_value'] = (
    (data['total_spent'] > 600) | 
    (data['visit_frequency'] > 6) | 
    (data['purchase_frequency'] > 0.7)
).astype(int)

df = pd.DataFrame(data)

# Explore the data
print(df.head())
print("\nHigh-value customer distribution:")
print(df['high_value'].value_counts())

# Feature engineering and preprocessing
X = df[['total_spent', 'visit_frequency', 'age', 'purchase_frequency']]
y = df['high_value']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Predict new customers
new_customers = np.array([
    [800, 8, 35, 0.9],  # Likely high-value
    [300, 3, 25, 0.2]   # Likely low-value
])
new_customers_scaled = scaler.transform(new_customers)

predictions = model.predict(new_customers_scaled)
probabilities = model.predict_proba(new_customers_scaled)

print("\nNew Customer Predictions:")
for i, (customer, pred, prob) in enumerate(zip(new_customers, predictions, probabilities)):
    print(f"\nCustomer {i+1} Features:")
    print(f"Total Spent: ${customer[0]:.2f}")
    print(f"Visit Frequency: {customer[1]} times/month")
    print(f"Age: {customer[2]} years")
    print(f"Purchase Frequency: {customer[3]:.1f}")
    print(f"Prediction: {'High Value' if pred == 1 else 'Low Value'}")
    print(f"Confidence: {prob[pred]*100:.1f}%")
    
    
    

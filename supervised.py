#housing




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and explore the dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
housing = pd.read_csv(url)
print(housing.head())

# Data preprocessing
housing = housing.dropna()  # Remove missing values
X = housing[['median_income', 'housing_median_age', 'population']]  # Features
y = housing['median_house_value']  # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"R-squared: {r2:.2f}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()

# Predict a new house's price
new_house = np.array([[4.5, 25, 2000]])  # income, age, population
predicted_price = model.predict(new_house)
print(f"\nPredicted price for new house: ${predicted_price[0]:,.2f}")
















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
    
    
    

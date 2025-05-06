#Bin Packing Problem (CSP)

from ortools.sat.python import cp_model

def solve_bin_packing():
    # Create model
    model = cp_model.CpModel()
    
    # Items and bins
    item_weights = [4, 8, 1, 4, 2, 6, 3, 5]
    num_items = len(item_weights)
    bins = [0, 1, 2]  # Three bins available
    bin_capacity = 12
    
    # Create variables: assignment[i] represents which bin item i goes to
    assignment = {}
    for i in range(num_items):
        assignment[i] = model.NewIntVar(0, len(bins)-1, f'item_{i}_bin')
    
    # Create bin loads variables
    bin_loads = {}
    for b in bins:
        bin_loads[b] = model.NewIntVar(0, bin_capacity, f'bin_{b}_load')
    
    # Add constraints
    # 1. Bin capacity constraints
    for b in bins:
        model.Add(
            sum(item_weights[i] * (assignment[i] == b) for i in range(num_items)) <= bin_capacity
        )
    
    # 2. Items 2 and 1 in different bins
    model.Add(assignment[2] != assignment[1])
    
    # 3. Items 0 and 5 in same bin
    model.Add(assignment[0] == assignment[5])
    
    # 4. Items 4 and 5 in different bins
    model.Add(assignment[4] != assignment[5])
    
    # 5. Items 6 and 7 in same bin, not with item 1
    model.Add(assignment[6] == assignment[7])
    model.Add(assignment[6] != assignment[1])
    
    # 6. Item 5 not in bin 0
    model.Add(assignment[5] != 0)
    
    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Print solution
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Optimal item assignment:")
        bin_contents = {b: [] for b in bins}
        bin_weights = {b: 0 for b in bins}
        
        for i in range(num_items):
            b = solver.Value(assignment[i])
            bin_contents[b].append(i)
            bin_weights[b] += item_weights[i]
        
        for b in bins:
            print(f"Bin {b}: Items {bin_contents[b]} (Total weight: {bin_weights[b]})")
        
        # Verify constraints
        print("\nConstraint Verification:")
        assignments = [solver.Value(assignment[i]) for i in range(num_items)]
        
        # 1. Bin capacities
        print("- All bin weights ≤ 12:", all(w <= bin_capacity for w in bin_weights.values()))
        
        # 2. Items 2 and 1 different bins
        print("- Items 2 and 1 different bins:", assignments[2] != assignments[1])
        
        # 3. Items 0 and 5 same bin
        print("- Items 0 and 5 same bin:", assignments[0] == assignments[5])
        
        # 4. Items 4 and 5 different bins
        print("- Items 4 and 5 different bins:", assignments[4] != assignments[5])
        
        # 5. Items 6 and 7 same bin, not with 1
        print("- Items 6 and 7 same bin:", assignments[6] == assignments[7])
        print("- Items 6 and 7 not with item 1:", assignments[6] != assignments[1])
        
        # 6. Item 5 not in bin 0
        print("- Item 5 not in bin 0:", assignments[5] != 0)
        
        print(f"\nSolver status: {'Optimal' if status == cp_model.OPTIMAL else 'Feasible'} solution found")
    else:
        print("No solution found")

solve_bin_packing()










#Amazon Sale Report (Amount Prediction)
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


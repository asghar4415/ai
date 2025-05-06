#Panelist Seating Arrangement (CSP)


from ortools.sat.python import cp_model

def solve_panelist_seating():
    # Create model
    model = cp_model.CpModel()
    
    # Panelists and seats
    panelists = ['Amir', 'Bella', 'Charles', 'Diana', 'Ethan', 'Farah']
    num_seats = 6
    seats = list(range(num_seats))
    
    # Create variables: seat[i] represents which panelist sits there
    seating = {}
    for seat in seats:
        seating[seat] = model.NewIntVar(0, len(panelists)-1, f'seat_{seat}')
    
    # Add constraints
    # 1. All panelists must be in different seats
    model.AddAllDifferent([seating[seat] for seat in seats])
    
    # 2. Bella must be left of Farah
    bella = panelists.index('Bella')
    farah = panelists.index('Farah')
    for i in seats:
        for j in seats:
            if i >= j:
                model.Add(seating[i] != bella).OnlyEnforceIf(seating[j] == farah)
    
    # 3. Charles must sit next to Diana
    charles = panelists.index('Charles')
    diana = panelists.index('Diana')
    adjacent_pairs = []
    for i in range(num_seats - 1):
        adjacent_pairs.append((i, i+1))
    for i in range(1, num_seats):
        adjacent_pairs.append((i, i-1))
    
    adjacent_constraints = []
    for i, j in adjacent_pairs:
        is_adjacent = model.NewBoolVar(f'adjacent_{i}_{j}')
        model.Add(seating[i] == charles).OnlyEnforceIf(is_adjacent)
        model.Add(seating[j] == diana).OnlyEnforceIf(is_adjacent)
        adjacent_constraints.append(is_adjacent)
    model.Add(sum(adjacent_constraints) >= 1)
    
    # 4. Amir cannot sit at ends
    amir = panelists.index('Amir')
    model.Add(seating[0] != amir)
    model.Add(seating[5] != amir)
    
    # 5. Ethan must be in middle (seat 2 or 3)
    ethan = panelists.index('Ethan')
    model.Add(seating[2] == ethan).OnlyEnforceIf(model.NewBoolVar('ethan_seat2'))
    model.Add(seating[3] == ethan).OnlyEnforceIf(model.NewBoolVar('ethan_seat3'))
    model.AddBoolOr([seating[2] == ethan, seating[3] == ethan])
    
    # 6. Diana cannot sit at ends
    model.Add(seating[0] != diana)
    model.Add(seating[5] != diana)
    
    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Print solution
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Optimal seating arrangement:")
        for seat in seats:
            panelist_index = solver.Value(seating[seat])
            print(f"Seat {seat}: {panelists[panelist_index]}")
        
        # Verify constraints
        print("\nConstraint Verification:")
        seating_arrangement = [panelists[solver.Value(seating[seat])] for seat in seats]
        
        # 1. All different
        print(f"- All panelists unique: {len(set(seating_arrangement)) == len(panelists)}")
        
        # 2. Bella left of Farah
        bella_pos = seating_arrangement.index('Bella')
        farah_pos = seating_arrangement.index('Farah')
        print(f"- Bella left of Farah: {bella_pos < farah_pos}")
        
        # 3. Charles next to Diana
        charles_pos = seating_arrangement.index('Charles')
        diana_pos = seating_arrangement.index('Diana')
        print(f"- Charles next to Diana: {abs(charles_pos - diana_pos) == 1}")
        
        # 4. Amir not at ends
        print(f"- Amir not at ends: {'Amir' not in [seating_arrangement[0], seating_arrangement[5]]}")
        
        # 5. Ethan in middle
        print(f"- Ethan in middle: {seating_arrangement[2] == 'Ethan' or seating_arrangement[3] == 'Ethan'}")
        
        # 6. Diana not at ends
        print(f"- Diana not at ends: {'Diana' not in [seating_arrangement[0], seating_arrangement[5]]}")
        
        print(f"\nSolver status: {'Optimal' if status == cp_model.OPTIMAL else 'Feasible'} solution found")
    else:
        print("No solution found")

solve_panelist_seating()







#Amazon Sale Report (B2B/B2C Classification)
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



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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def amazon_sale_analysis():
    # Part 1: Data Pre-processing
    # Load dataset (assuming file is named 'amazon_sales.csv')
    try:
        df = pd.read_csv('amazon_sales.csv')
    except FileNotFoundError:
        print("Error: File 'amazon_sales.csv' not found")
        return
    
    print("\n=== Part 1: Data Pre-processing ===")
    print("\nInitial data shape:", df.shape)
    print("\nInitial data info:")
    print(df.info())
    
    # Handle missing values
    print("\nMissing values before handling:")
    print(df.isnull().sum())
    
    # Separate features and target
    X = df.drop('B2B', axis=1)
    y = df['B2B']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    print("\nCategorical columns:", list(categorical_cols))
    print("Numerical columns:", list(numerical_cols))
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])
    
    # Part 2: Data Splitting and Model Training
    print("\n=== Part 2: Data Splitting and Model Training ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    # Train and evaluate models
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', model)])
        
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        
        # Store the trained model back
        models[name] = pipeline
    
    # Part 3: Model Evaluation
    print("\n=== Part 3: Model Evaluation ===")
    
    # Evaluate on training data
    print("\nTraining Set Performance:")
    for name, model in models.items():
        y_train_pred = model.predict(X_train)
        print(f"\n{name}:")
        print("Accuracy:", accuracy_score(y_train, y_train_pred))
        print("Classification Report:")
        print(classification_report(y_train, y_train_pred))
    
    # Evaluate on test data
    print("\nTest Set Performance:")
    for name, model in models.items():
        y_test_pred = model.predict(X_test)
        print(f"\n{name}:")
        print("Accuracy:", accuracy_score(y_test, y_test_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_test_pred))
    
    # Cross-validation
    print("\nCross-Validation Results:")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        print(f"\n{name} CV Accuracy: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
    
    # Return best model based on test accuracy
    best_model_name = max(models.keys(), 
                         key=lambda x: accuracy_score(y_test, models[x].predict(X_test)))
    print(f"\nBest model: {best_model_name}")
    
    return models[best_model_name]

# Run the analysis
best_model = amazon_sale_analysis()

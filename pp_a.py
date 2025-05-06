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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def amazon_sale_regression():
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
    X = df.drop('Amount', axis=1)
    y = df['Amount']
    
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
        'Random Forest': RandomForestRegressor(random_state=42),
        'SVR': SVR(),
        'Linear Regression': LinearRegression()
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
        print("R² Score:", r2_score(y_train, y_train_pred))
        print("Mean Absolute Error:", mean_absolute_error(y_train, y_train_pred))
        print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
    
    # Evaluate on test data
    print("\nTest Set Performance:")
    for name, model in models.items():
        y_test_pred = model.predict(X_test)
        print(f"\n{name}:")
        print("R² Score:", r2_score(y_test, y_test_pred))
        print("Mean Absolute Error:", mean_absolute_error(y_test, y_test_pred))
        print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
    
    # Cross-validation
    print("\nCross-Validation Results:")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
        print(f"\n{name} CV R²: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
    
    # Return best model based on test R²
    best_model_name = max(models.keys(), 
                         key=lambda x: r2_score(y_test, models[x].predict(X_test)))
    print(f"\nBest model: {best_model_name}")
    
    return models[best_model_name]

# Run the analysis
best_regressor = amazon_sale_regression()




from helper_function import * 

def main():
    # Step 1: Load data
    filepath = "boston.csv"
    print("Loading Dataset...")  
    data = load_data(filepath)

    # Step 2: Split data into train and test sets
    target_column = "MEDV" 
    print("Splitting Dataset...")
    X_train, X_test, y_train, y_test = split_data(data, target_column)

    # Step 3: Train and evaluate multiple models
    # Train and evaluate multiple models, returning results as a DataFrame
    models_to_train = ['linear', 'ridge', 'lasso', 'random_forest']
    results_df = pd.DataFrame()
    print("Training and Evaluating models...")

    # Step 4: Train and evaluate models
    for algorithm in models_to_train:
        print(f"Training and evaluating {algorithm} model...")
        model, metrics_df = train_and_evaluate_model_df(
        algorithm, X_train, y_train, X_test, y_test,
        alpha=0.1, n_estimators=100, random_state=42
    )
    
        # Concatenate only the metrics DataFrame
        results_df = pd.concat([results_df, metrics_df], ignore_index=True)

    # Step 4: Analyze feature importance
    feature_names = X_train.columns
    importance_df = feature_importance(model, feature_names)
    print("\nFeature Importances:")
    print(importance_df.head(5))

    print("\nModel Evaluation Results:")
    print(results_df)

if __name__ == "__main__":
    main()

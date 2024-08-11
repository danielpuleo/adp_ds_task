from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

def kfold_cross_validation(X, y, model, n_splits=5, threshold=0.5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=12)
    
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []
    log_losses = []
    
    fold = 1
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # For models that support predict_proba
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            
            # Apply the custom threshold for binary classification
            if probabilities.shape[1] == 2:  # Binary classification
                predictions = (probabilities[:, 1] >= threshold).astype(int)
            else:
                # For multi-class, assign class if the max probability exceeds the threshold
                predictions = np.argmax(probabilities, axis=1)
                max_probabilities = np.max(probabilities, axis=1)
                predictions = np.where(max_probabilities >= threshold, predictions, -1)  # -1 means "no confident prediction"
        else:
            # For models that don't support probabilities, use predict
            predictions = model.predict(X_test)
        
        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        
        # Calculate other metrics
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted',zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted',zero_division=0)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        if hasattr(model, "predict_proba"):
            log_loss_value = log_loss(y_test, probabilities)
            log_losses.append(log_loss_value)
        
        fold += 1
    
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    
    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Mean Precision: {mean_precision}")
    print(f"Mean Recall: {mean_recall}")
    print(f"Mean F1 Score: {mean_f1}")
    
    if log_losses:
        mean_log_loss = np.mean(log_losses)
        print(f"Mean Log Loss: {mean_log_loss}")
    
    return {
        "accuracy": mean_accuracy,
        "precision": mean_precision,
        "recall": mean_recall,
        "f1": mean_f1,
        "log_loss": np.mean(log_losses) if log_losses else None
    }
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
import joblib

class OptimalBankingModel:
    def __init__(self, confidence_threshold=0.85):
        """
        Optimal Banking Model: GradientBoosting with 85% confidence threshold
        Based on analysis showing best balance of precision (90.7%) and coverage (36.1%)
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_optimized_pipeline(self):
        """Create the optimized GradientBoosting pipeline"""
        
        # Based on your cross-validation results, these are good hyperparameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200,           # More trees for stability
                learning_rate=0.1,          # Balanced learning rate
                max_depth=5,               # Sufficient complexity
                min_samples_split=2,       # Standard splitting
                min_samples_leaf=1,        # Allow fine-grained splits
                random_state=42,
                verbose=0
            ))
        ])
        
        return pipeline
    
    def train_with_hyperparameter_tuning(self, X, y):
        """Train model with hyperparameter optimization"""
        
        print("Training Optimal Banking Model...")
        print("Model: GradientBoostingClassifier")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"Training Data: {len(X)} samples, {len(X.columns)} features")
        
        # Create base pipeline
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(random_state=42, verbose=0))
        ])
        
        # Hyperparameter grid (focused on best performing ranges)
        param_grid = {
            'clf__n_estimators': [150, 200, 250],
            'clf__learning_rate': [0.05, 0.1, 0.15],
            'clf__max_depth': [4, 5, 6],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2]
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X, y)
        
        # Store the best model
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_trained = True
        
        print(f"\nBest Parameters Found:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        print(f"Best CV Score (ROC-AUC): {grid_search.best_score_:.4f}")
        
        return self
    
    def predict_with_confidence(self, X):
        """Make predictions with confidence filtering"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained first. Call train_with_hyperparameter_tuning()")
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Determine high-confidence predictions
        high_confidence_mask = (
            (probabilities >= self.confidence_threshold) | 
            (probabilities <= (1 - self.confidence_threshold))
        )
        
        # Create prediction array (default to -1 for uncertain cases)
        predictions = np.full(len(X), -1)  # -1 means "requires human review"
        
        # Only predict for high-confidence cases
        confident_indices = np.where(high_confidence_mask)[0]
        confident_probabilities = probabilities[confident_indices]
        confident_predictions = (confident_probabilities >= 0.5).astype(int)
        
        predictions[confident_indices] = confident_predictions
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confident_mask': high_confidence_mask,
            'coverage': np.sum(high_confidence_mask) / len(X)
        }
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        
        # Get predictions with confidence
        results = self.predict_with_confidence(X_test)
        
        predictions = results['predictions']
        probabilities = results['probabilities']
        confident_mask = results['confident_mask']
        coverage = results['coverage']
        
        # Extract confident predictions only
        confident_indices = confident_mask
        if np.sum(confident_indices) == 0:
            return {
                'coverage': 0.0,
                'confident_accuracy': 0.0,
                'confident_samples': 0,
                'total_samples': len(X_test),
                'error': 'No confident predictions made'
            }
        
        y_confident = y_test[confident_indices]
        pred_confident = predictions[confident_indices]
        prob_confident = probabilities[confident_indices]
        
        # Calculate metrics for confident predictions only
        confident_accuracy = accuracy_score(y_confident, pred_confident)
        confident_balanced_accuracy = balanced_accuracy_score(y_confident, pred_confident)
        
        # ROC-AUC
        if len(np.unique(y_confident)) > 1:
            confident_auc = roc_auc_score(y_confident, prob_confident)
        else:
            confident_auc = np.nan
        
        # Confusion matrix and cost
        cm = confusion_matrix(y_confident, pred_confident)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            cost_5xFP_1xFN = 5 * fp + 1 * fn
        else:
            fp = fn = cost_5xFP_1xFN = np.nan
        
        return {
            'coverage': coverage,
            'confident_accuracy': confident_accuracy,
            'confident_balanced_accuracy': confident_balanced_accuracy,
            'confident_auc': confident_auc,
            'confident_samples': np.sum(confident_indices),
            'total_samples': len(X_test),
            'cost_5xFP_1xFN': cost_5xFP_1xFN,
            'false_positives': fp,
            'false_negatives': fn,
            'confusion_matrix': cm
        }
    
    def cross_validate_performance(self, X, y, cv_folds=5):
        """Cross-validation to verify expected performance"""
        
        print(f"\nCross-Validation Performance Assessment")
        print("-" * 50)
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model for this fold
            temp_model = OptimalBankingModel(self.confidence_threshold)
            temp_model.train_with_hyperparameter_tuning(X_train, y_train)
            
            # Evaluate
            fold_results = temp_model.evaluate_model(X_test, y_test.values)
            fold_results['fold'] = fold
            cv_results.append(fold_results)
            
            print(f"Fold {fold}: Accuracy={fold_results['confident_accuracy']:.3f}, "
                  f"Coverage={fold_results['coverage']:.3f}, "
                  f"Samples={fold_results['confident_samples']}/{fold_results['total_samples']}")
        
        # Summary statistics
        cv_df = pd.DataFrame(cv_results)
        
        summary = {
            'mean_accuracy': cv_df['confident_accuracy'].mean(),
            'std_accuracy': cv_df['confident_accuracy'].std(),
            'mean_coverage': cv_df['coverage'].mean(),
            'std_coverage': cv_df['coverage'].std(),
            'mean_cost': cv_df['cost_5xFP_1xFN'].mean(),
            'mean_samples_per_fold': cv_df['confident_samples'].mean()
        }
        
        print(f"\nCross-Validation Summary:")
        print(f"  Confident Accuracy: {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")
        print(f"  Coverage: {summary['mean_coverage']:.3f} ± {summary['std_coverage']:.3f}")
        print(f"  Average Cost: {summary['mean_cost']:.1f}")
        print(f"  Average Samples per Fold: {summary['mean_samples_per_fold']:.0f}")
        
        return cv_results, summary
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        joblib.dump({
            'model': self.model,
            'confidence_threshold': self.confidence_threshold,
            'best_params': self.best_params
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        saved_data = joblib.load(filepath)
        
        self.model = saved_data['model']
        self.confidence_threshold = saved_data['confidence_threshold']
        self.best_params = saved_data['best_params']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
        print(f"Confidence Threshold: {self.confidence_threshold}")
    
    def get_business_summary(self, X, y):
        """Generate business-friendly summary"""
        
        if not self.is_trained:
            self.train_with_hyperparameter_tuning(X, y)
        
        # Get overall performance
        eval_results = self.evaluate_model(X, y.values)
        
        total_applications = len(X)
        automated_applications = int(total_applications * eval_results['coverage'])
        manual_applications = total_applications - automated_applications
        expected_accuracy = eval_results['confident_accuracy']
        
        print(f"\nBUSINESS IMPACT SUMMARY")
        print("=" * 50)
        print(f"Model Type: GradientBoosting with {self.confidence_threshold*100}% Confidence")
        print(f"Training Data: {total_applications:,} applications")
        print(f"\nAutomated Processing:")
        print(f"  • {automated_applications:,} applications ({eval_results['coverage']:.1%})")
        print(f"  • Expected accuracy: {expected_accuracy:.1%}")
        print(f"  • Cost score: {eval_results['cost_5xFP_1xFN']:.1f}")
        print(f"\nHuman Review Required:")
        print(f"  • {manual_applications:,} applications ({1-eval_results['coverage']:.1%})")
        print(f"\nRisk Assessment:")
        print(f"  • False Positives: {eval_results['false_positives']}")
        print(f"  • False Negatives: {eval_results['false_negatives']}")
        print(f"  • Banking Suitable: {'Yes' if expected_accuracy >= 0.85 else 'No'}")

# Ready-to-use implementation
def train_optimal_banking_model(df):
    """Complete workflow to train the optimal banking model"""
    
    # Prepare data
    y = (df['target'] == 1).astype(int)
    X = df.drop(columns=['target']).copy()
    
    print("OPTIMAL BANKING MODEL TRAINING")
    print("=" * 50)
    print("Configuration: GradientBoosting + 85% Confidence")
    print("Expected: 90.7% Accuracy, 36.1% Coverage")
    
    # Create and train model
    banking_model = OptimalBankingModel(confidence_threshold=0.85)
    banking_model.train_with_hyperparameter_tuning(X, y)
    
    # Verify performance
    cv_results, summary = banking_model.cross_validate_performance(X, y)
    
    # Business summary
    banking_model.get_business_summary(X, y)
    
    return banking_model, cv_results, summary

# Execute training
df = pd.read_csv("kredit_final.csv")  # Load your cleaned dataset here
print("Starting optimal banking model training...")
optimal_model, cv_results, performance_summary = train_optimal_banking_model(df)

print(f"\nModel ready! Use optimal_model.predict_with_confidence(new_data) for predictions.")
print(f"To save: optimal_model.save_model('banking_model.pkl')")
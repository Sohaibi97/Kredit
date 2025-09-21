import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

class BankingEnsembleModel:
    def __init__(self, confidence_threshold=0.7):
        """
        Ensemble of LogisticRegression + GradientBoosting with confidence-based predictions
        """
        self.confidence_threshold = confidence_threshold
        self.models = {}
        self.best_params = {}
        
    def get_optimized_model_configs(self):
        """Define model configurations with optimized hyperparameters from previous results"""
        return {
            'LogisticRegression': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(random_state=42, max_iter=1000))
                ]),
                'param_grid': {
                    'clf__C': [0.1],  # Best from previous results
                    'clf__penalty': ['l2'],
                    'clf__solver': ['liblinear']
                }
            },
            'GradientBoosting': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', GradientBoostingClassifier(random_state=42))
                ]),
                'param_grid': {
                    'clf__n_estimators': [100, 200],
                    'clf__learning_rate': [0.05, 0.1],
                    'clf__max_depth': [3, 5],
                    'clf__min_samples_split': [2, 5],
                    'clf__min_samples_leaf': [1, 2]
                }
            }
        }
    
    def train_individual_models(self, X_train, y_train):
        """Train individual models and store best parameters"""
        model_configs = self.get_optimized_model_configs()
        trained_models = {}
        
        for model_name, config in model_configs.items():
            print(f"Training {model_name}...")
            
            # Grid search for best parameters
            grid_search = GridSearchCV(
                config['pipeline'],
                config['param_grid'],
                cv=3,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            trained_models[model_name] = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
            
            print(f"  Best params for {model_name}: {grid_search.best_params_}")
        
        return trained_models
    
    def get_ensemble_predictions(self, models, X_test):
        """Get ensemble predictions using different strategies"""
        predictions = {}
        probabilities = {}
        
        # Get individual model predictions
        for model_name, model in models.items():
            probas = model.predict_proba(X_test)[:, 1]
            predictions[model_name] = probas
            probabilities[model_name] = probas
        
        # Ensemble strategies
        ensemble_results = {}
        
        # 1. Simple Average
        avg_probas = np.mean(list(probabilities.values()), axis=0)
        ensemble_results['SimpleAverage'] = avg_probas
        
        # 2. Weighted Average (LogisticRegression gets higher weight due to better accuracy)
        lr_probas = probabilities['LogisticRegression']
        gb_probas = probabilities['GradientBoosting']
        weighted_probas = 0.6 * lr_probas + 0.4 * gb_probas  # Weight based on accuracy
        ensemble_results['WeightedAverage'] = weighted_probas
        
        # 3. Consensus (Conservative) - both models must be confident and agree
        consensus_probas = np.full(len(X_test), 0.5)  # Default to uncertain
        consensus_confident = np.zeros(len(X_test), dtype=bool)
        
        for i in range(len(X_test)):
            lr_conf = (lr_probas[i] >= self.confidence_threshold) or (lr_probas[i] <= (1 - self.confidence_threshold))
            gb_conf = (gb_probas[i] >= self.confidence_threshold) or (gb_probas[i] <= (1 - self.confidence_threshold))
            
            if lr_conf and gb_conf:
                # Both confident - check if they agree
                lr_pred = 1 if lr_probas[i] >= 0.5 else 0
                gb_pred = 1 if gb_probas[i] >= 0.5 else 0
                
                if lr_pred == gb_pred:  # Agreement
                    consensus_probas[i] = np.mean([lr_probas[i], gb_probas[i]])
                    consensus_confident[i] = True
        
        ensemble_results['Consensus'] = consensus_probas
        ensemble_results['consensus_mask'] = consensus_confident
        
        return ensemble_results, probabilities
    
    def evaluate_ensemble_with_confidence(self, ensemble_probas, y_test, ensemble_name, consensus_mask=None):
        """Evaluate ensemble with confidence-based approach"""
        
        if consensus_mask is not None:
            # For consensus, use the consensus mask
            high_confidence_mask = consensus_mask
        else:
            # For other ensembles, use standard confidence threshold
            high_confidence_mask = (ensemble_probas >= self.confidence_threshold) | (ensemble_probas <= (1 - self.confidence_threshold))
        
        if np.sum(high_confidence_mask) == 0:
            return {
                'ensemble_name': ensemble_name,
                'coverage': 0.0,
                'confident_accuracy': 0.0,
                'confident_balanced_accuracy': 0.0,
                'confident_auc': 0.0,
                'total_samples': len(y_test),
                'confident_samples': 0,
                'cost': np.nan,
                'FP': np.nan,
                'FN': np.nan
            }
        
        # Get confident predictions
        y_confident = y_test[high_confidence_mask]
        probas_confident = ensemble_probas[high_confidence_mask]
        
        # Make predictions for confident cases
        predictions_confident = (probas_confident >= 0.5).astype(int)
        
        # Calculate metrics
        coverage = np.sum(high_confidence_mask) / len(y_test)
        confident_accuracy = accuracy_score(y_confident, predictions_confident)
        confident_balanced_accuracy = balanced_accuracy_score(y_confident, predictions_confident)
        
        # AUC only if we have both classes
        if len(np.unique(y_confident)) > 1:
            confident_auc = roc_auc_score(y_confident, probas_confident)
        else:
            confident_auc = np.nan
        
        # Cost calculation
        if len(y_confident) > 0:
            tn, fp, fn, tp = confusion_matrix(y_confident, predictions_confident).ravel()
            cost_5xFP_1xFN = 5 * fp + 1 * fn
        else:
            fp = fn = cost_5xFP_1xFN = np.nan
        
        return {
            'ensemble_name': ensemble_name,
            'coverage': coverage,
            'confident_accuracy': confident_accuracy,
            'confident_balanced_accuracy': confident_balanced_accuracy,
            'confident_auc': confident_auc,
            'total_samples': len(y_test),
            'confident_samples': np.sum(high_confidence_mask),
            'cost': cost_5xFP_1xFN,
            'FP': fp,
            'FN': fn
        }
    
    def cross_validate_ensemble(self, X, y, cv_folds=5):
        """Cross-validation for ensemble models"""
        
        results = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        print(f"🏦 BANKING ENSEMBLE ANALYSIS (Confidence Threshold: {self.confidence_threshold})")
        print("=" * 70)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n--- FOLD {fold} ---")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train individual models
            models = self.train_individual_models(X_train, y_train)
            
            # Get ensemble predictions
            ensemble_results, individual_probas = self.get_ensemble_predictions(models, X_test)
            
            # Evaluate each ensemble strategy
            fold_results = []
            
            for ensemble_name, probas in ensemble_results.items():
                if ensemble_name == 'consensus_mask':
                    continue
                
                consensus_mask = ensemble_results.get('consensus_mask') if ensemble_name == 'Consensus' else None
                
                result = self.evaluate_ensemble_with_confidence(
                    probas, y_test.values, ensemble_name, consensus_mask
                )
                result['fold'] = fold
                fold_results.append(result)
            
            # Also evaluate individual models for comparison
            for model_name, probas in individual_probas.items():
                result = self.evaluate_ensemble_with_confidence(
                    probas, y_test.values, f"Individual_{model_name}"
                )
                result['fold'] = fold
                fold_results.append(result)
            
            # Print fold summary
            print(f"\nFold {fold} Results:")
            for result in fold_results:
                name = result['ensemble_name']
                acc = result['confident_accuracy']
                cov = result['coverage']
                samples = result['confident_samples']
                total = result['total_samples']
                print(f"  {name:20s}: Accuracy={acc:.3f}, Coverage={cov:.3f}, Samples={samples}/{total}")
            
            results.extend(fold_results)
        
        return pd.DataFrame(results)

def run_ensemble_analysis(df):
    """Run complete ensemble analysis"""
    
    # Prepare data
    y = (df['target'] == 1).astype(int)
    X = df.drop(columns=['target']).copy()
    
    print("📊 DATA PREPARATION:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target distribution: {y.sum()} good ({y.mean():.1%}) / {len(y) - y.sum()} bad ({1-y.mean():.1%})")
    
    # Run ensemble analysis
    ensemble_model = BankingEnsembleModel(confidence_threshold=0.7)
    results_df = ensemble_model.cross_validate_ensemble(X, y)
    
    # Summary analysis
    print(f"\n{'='*70}")
    print("🏆 ENSEMBLE SUMMARY RESULTS")
    print("=" * 70)
    
    summary = results_df.groupby('ensemble_name').agg({
        'coverage': 'mean',
        'confident_accuracy': 'mean',
        'confident_balanced_accuracy': 'mean',
        'confident_auc': 'mean',
        'confident_samples': 'mean',
        'cost': 'mean',
        'FP': 'mean',
        'FN': 'mean'
    }).round(3)
    
    print(summary)
    
    # Find best ensemble
    banking_suitable = summary[
        (summary['confident_accuracy'] >= 0.80) & 
        (summary['coverage'] >= 0.25)
    ].copy()
    
    if len(banking_suitable) > 0:
        print(f"\n✅ BANKING-SUITABLE ENSEMBLES:")
        banking_suitable_sorted = banking_suitable.sort_values(
            ['confident_accuracy', 'coverage'], 
            ascending=[False, False]
        )
        print(banking_suitable_sorted)
        
        # Best ensemble
        best_ensemble = banking_suitable_sorted.index[0]
        best_stats = banking_suitable_sorted.iloc[0]
        
        print(f"\n🥇 BEST ENSEMBLE: {best_ensemble}")
        print(f"   Confident Accuracy: {best_stats['confident_accuracy']:.3f}")
        print(f"   Coverage: {best_stats['coverage']:.3f}")
        print(f"   AUC: {best_stats['confident_auc']:.3f}")
        print(f"   Cost Score: {best_stats['cost']:.1f}")
        print(f"   Average samples per fold: {best_stats['confident_samples']:.0f}")
        
        coverage_pct = best_stats['coverage'] * 100
        auto_cases = int(len(df) * best_stats['coverage'])
        manual_cases = len(df) - auto_cases
        
        print(f"\n📊 PRACTICAL IMPACT:")
        print(f"   • {auto_cases:,} cases automated ({coverage_pct:.1f}%)")
        print(f"   • {manual_cases:,} cases for human review")
        print(f"   • Expected accuracy on automated cases: {best_stats['confident_accuracy']:.1%}")
        
        # Compare with individual models
        individual_results = summary[summary.index.str.startswith('Individual_')]
        if len(individual_results) > 0:
            print(f"\n📈 IMPROVEMENT OVER INDIVIDUAL MODELS:")
            for idx, row in individual_results.iterrows():
                model_name = idx.replace('Individual_', '')
                acc_improvement = best_stats['confident_accuracy'] - row['confident_accuracy']
                cov_change = best_stats['coverage'] - row['coverage']
                print(f"   vs {model_name}:")
                print(f"     Accuracy: {acc_improvement:+.3f} ({acc_improvement/row['confident_accuracy']*100:+.1f}%)")
                print(f"     Coverage: {cov_change:+.3f} ({cov_change/row['coverage']*100:+.1f}%)")
    else:
        print("❌ No ensemble meets banking standards")
        print("Best individual model results from previous analysis:")
        print("LogisticRegression: 85.6% accuracy, 43.9% coverage")
    
    return results_df

# Execute the ensemble analysis
print("🚀 Starting Banking Ensemble Analysis...")
ensemble_results = run_ensemble_analysis(df)
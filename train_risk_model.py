"""
Training Script for Fire App Risk Assessment Model
Trains and evaluates ML models for wildfire risk prediction
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.risk_model import PowerLineRiskModel
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)


def plot_training_results(model, results):
    """Create visualization of training results."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # === CLASSIFICATION RESULTS ===
    if 'classifier' in results:
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        y_true = results['classifier']['true_labels']
        y_pred = results['classifier']['predictions']
        
        cm = confusion_matrix(y_true, y_pred)
        labels = model.label_encoder.inverse_transform(np.unique(y_true))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Classification Metrics
        ax2 = plt.subplot(2, 3, 2)
        metrics = model.metrics['classifier']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics['accuracy'], metrics['precision'],
            metrics['recall'], metrics['f1_score']
        ]
        
        bars = ax2.bar(metric_names, metric_values,
                      color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        ax2.set_ylim([0, 1])
        ax2.set_title('Classification Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Score')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Feature Importance (Classifier)
        ax3 = plt.subplot(2, 3, 3)
        if 'classifier' in model.feature_importance:
            feat_imp = model.feature_importance['classifier']
            # Get top 10
            top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
            features = [f[0] for f in top_features]
            values = [f[1] for f in top_features]
            
            ax3.barh(features, values, color='steelblue')
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 10 Features (Classifier)', fontsize=14, fontweight='bold')
            ax3.invert_yaxis()
    
    # === REGRESSION RESULTS ===
    if 'regressor' in results:
        # 4. Predicted vs Actual
        ax4 = plt.subplot(2, 3, 4)
        y_true_reg = results['regressor']['true_scores']
        y_pred_reg = results['regressor']['predictions']
        
        ax4.scatter(y_true_reg, y_pred_reg, alpha=0.5, s=30, color='coral')
        ax4.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
        ax4.set_xlabel('True Risk Score')
        ax4.set_ylabel('Predicted Risk Score')
        ax4.set_title('Predicted vs Actual Risk Scores', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Residuals Distribution
        ax5 = plt.subplot(2, 3, 5)
        residuals = y_true_reg - y_pred_reg
        
        ax5.hist(residuals, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax5.axvline(0, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Residual (True - Predicted)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax5.text(0.05, 0.95, f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}',
                transform=ax5.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. Feature Importance (Regressor)
        ax6 = plt.subplot(2, 3, 6)
        if 'regressor' in model.feature_importance:
            feat_imp = model.feature_importance['regressor']
            # Get top 10
            top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
            features = [f[0] for f in top_features]
            values = [f[1] for f in top_features]
            
            ax6.barh(features, values, color='coral')
            ax6.set_xlabel('Importance')
            ax6.set_title('Top 10 Features (Regressor)', fontsize=14, fontweight='bold')
            ax6.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('fire_app_risk_model_training_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Training visualization saved as 'fire_app_risk_model_training_results.png'")
    plt.show()


def demonstrate_predictions(model):
    """Demonstrate model predictions on sample scenarios."""
    
    print("\n" + "="*70)
    print("🔮 SAMPLE RISK PREDICTIONS")
    print("="*70)
    
    scenarios = [
        {
            'name': '🟢 Low Risk Zone',
            'description': 'Low vegetation, good clearance, cool weather',
            'data': {
                'vegetation_height': 0.3,
                'clearance': 10.0,
                'ndvi': 0.25,
                'temperature': 18.0,
                'humidity': 65.0,
                'wind_speed': 3.0,
                'line_voltage': 115.0,
                'line_age': 15,
                'month': 3
            }
        },
        {
            'name': '🟡 Moderate Risk Zone',
            'description': 'Moderate vegetation, adequate clearance',
            'data': {
                'vegetation_height': 0.8,
                'clearance': 6.0,
                'ndvi': 0.5,
                'temperature': 26.0,
                'humidity': 45.0,
                'wind_speed': 7.0,
                'line_voltage': 230.0,
                'line_age': 30,
                'month': 6
            }
        },
        {
            'name': '🟠 High Risk Zone',
            'description': 'Dense vegetation, reduced clearance, hot & dry',
            'data': {
                'vegetation_height': 1.8,
                'clearance': 4.0,
                'ndvi': 0.65,
                'temperature': 34.0,
                'humidity': 25.0,
                'wind_speed': 12.0,
                'line_voltage': 345.0,
                'line_age': 45,
                'month': 8
            }
        },
        {
            'name': '🔴 Critical Risk Zone',
            'description': 'Very dense vegetation, minimal clearance, extreme conditions',
            'data': {
                'vegetation_height': 3.5,
                'clearance': 2.0,
                'ndvi': 0.75,
                'temperature': 40.0,
                'humidity': 15.0,
                'wind_speed': 20.0,
                'line_voltage': 500.0,
                'line_age': 60,
                'month': 8
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'─'*70}")
        print(f"Scenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'─'*70}")
        
        # Make prediction
        prediction = model.predict(scenario['data'])
        
        # Display results
        if 'classification' in prediction:
            print(f"\n🎯 CLASSIFICATION:")
            print(f"   Risk Level: {prediction['classification']['risk_level']}")
            print(f"   Confidence: {prediction['classification']['confidence']:.2%}")
            print(f"\n   Probabilities:")
            for level, prob in prediction['classification']['probabilities'].items():
                bar_length = int(prob * 30)
                bar = '█' * bar_length + '░' * (30 - bar_length)
                print(f"      {level:10s} {bar} {prob:.2%}")
        
        if 'regression' in prediction:
            print(f"\n📊 REGRESSION:")
            print(f"   Risk Score: {prediction['regression']['risk_score']:.4f}")
            print(f"   Risk %:     {prediction['regression']['risk_percentage']:.1f}%")
            
            # Visual risk meter
            score = prediction['regression']['risk_score']
            meter_length = 50
            filled = int(score * meter_length)
            
            if score < 0.3:
                color_symbol = '🟢'
            elif score < 0.6:
                color_symbol = '🟡'
            elif score < 0.85:
                color_symbol = '🟠'
            else:
                color_symbol = '🔴'
            
            meter = '█' * filled + '░' * (meter_length - filled)
            print(f"\n   Risk Meter: {color_symbol}")
            print(f"   [0%] {meter} [100%]")


def main():
    """Main training pipeline."""
    
    print("\n" + "="*70)
    print("🔥 FIRE APP - RISK ASSESSMENT MODEL TRAINING")
    print("   ML-Based Wildfire Risk Prediction for Power Lines")
    print("="*70)
    
    # === STEP 1: Initialize Model ===
    print("\n📌 Step 1: Initializing model...")
    model = PowerLineRiskModel(model_type='both')
    print("✅ Model initialized")
    
    # === STEP 2: Generate Training Data ===
    print("\n📌 Step 2: Generating training data...")
    features_df, risk_labels, risk_scores = model.generate_training_data(n_samples=3000)
    print(f"✅ Generated {len(features_df)} training samples")
    
    # === STEP 3: Train Models ===
    print("\n📌 Step 3: Training models...")
    results = model.train_models(features_df, risk_labels, risk_scores)
    
    # === STEP 4: Print Classification Report ===
    if 'classifier' in results:
        print("\n📊 DETAILED CLASSIFICATION REPORT:")
        y_true = results['classifier']['true_labels']
        y_pred = results['classifier']['predictions']
        labels = model.label_encoder.inverse_transform(np.unique(y_true))
        
        print(classification_report(y_true, y_pred, target_names=labels))
    
    # === STEP 5: Visualize Results ===
    print("\n📌 Step 4: Creating visualizations...")
    plot_training_results(model, results)
    
    # === STEP 6: Demonstrate Predictions ===
    demonstrate_predictions(model)
    
    # === STEP 7: Save Models ===
    print("\n📌 Step 5: Saving trained models...")
    
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    
    timestamp = model.save_models(filepath_prefix='models/fire_app_risk_model')
    
    if timestamp:
        print(f"✅ Models saved successfully!")
        print(f"\n📦 Saved files:")
        print(f"   - models/fire_app_risk_model_classifier_{timestamp}.pkl")
        print(f"   - models/fire_app_risk_model_regressor_{timestamp}.pkl")
        print(f"   - models/fire_app_risk_model_scaler_{timestamp}.pkl")
        print(f"   - models/fire_app_risk_model_label_encoder_{timestamp}.pkl")
        print(f"   - models/fire_app_risk_model_metadata_{timestamp}.json")
    
    # === SUMMARY ===
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("="*70)
    print("\n📊 Summary:")
    print(f"   • Trained on {len(features_df)} samples")
    print(f"   • Features: {len(features_df.columns)}")
    print(f"   • Models: Random Forest + Gradient Boosting")
    
    if 'classifier' in model.metrics:
        print(f"\n   🎯 Best Classifier: {model.metrics['classifier']['best_model']}")
        print(f"      Accuracy: {model.metrics['classifier']['accuracy']:.4f}")
    
    if 'regressor' in model.metrics:
        print(f"\n   📈 Best Regressor: {model.metrics['regressor']['best_model']}")
        print(f"      R² Score: {model.metrics['regressor']['r2_score']:.4f}")
    
    print("\n💡 Next Steps:")
    print("   1. Review 'fire_app_risk_model_training_results.png'")
    print("   2. Model is ready to use in your Flask app")
    print("   3. Run the app and check /api/risk_prediction endpoint")
    
    print("\n" + "="*70)
    
    return model, results


if __name__ == "__main__":
    model, results = main()
    print("\n✨ Training complete! Model is ready for deployment.")


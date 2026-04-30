import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

def run_model(X_train, X_test, y_train, y_test, feature_names):
    print("\n--- Random Forest (n_estimators=100) ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf, target_names=['Gerçek (0)', 'Sahte (1)']))

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n--- Özellik Önemi (Feature Importances) ---")
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
    plt.title('Random Forest - Özellik Önemi (Feature Importance)', fontweight='bold', fontsize=14)
    plt.xlabel('Önem Derecesi (Bilgi Kazancı)', fontsize=12)
    plt.ylabel('Özellik (Feature)', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Özellik önemi grafiği 'feature_importance.png' olarak kaydedildi.")

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Tahmin: 0', 'Tahmin: 1'],
                yticklabels=['Asıl: 0', 'Asıl: 1'], annot_kws={"size": 14})
    plt.title(f'Random Forest\nDoğruluk: {acc_rf*100:.2f}%', fontweight='bold', fontsize=13)

    plt.tight_layout()
    plt.savefig('confusion_matrix_random_forest.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Random Forest Karmaşıklık matrisi 'confusion_matrix_random_forest.png' olarak kaydedildi.")

    return {
        'name': 'Random Forest',
        'accuracy': acc_rf,
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'y_pred_proba': rf_model.predict_proba(X_test)[:, 1]
    }

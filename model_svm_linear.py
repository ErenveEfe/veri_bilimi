import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

def run_model(X_train, X_test, y_train, y_test):
    print("\n--- SVM (Kernel: Linear) ---")
    svm_linear = SVC(kernel='linear', random_state=42, probability=True)
    svm_linear.fit(X_train, y_train)
    y_pred_svm_linear = svm_linear.predict(X_test)

    acc_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
    print(f"SVM (Linear) Accuracy: {acc_svm_linear:.4f} ({acc_svm_linear*100:.2f}%)")
    print("\nClassification Report (Linear):")
    print(classification_report(y_test, y_pred_svm_linear, target_names=['Gerçek (0)', 'Sahte (1)']))

    cm_svm_linear = confusion_matrix(y_test, y_pred_svm_linear)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_svm_linear, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Tahmin: 0', 'Tahmin: 1'],
                yticklabels=['Asıl: 0', 'Asıl: 1'], annot_kws={"size": 14})
    plt.title(f'SVM (Linear)\nDoğruluk: {acc_svm_linear*100:.2f}%', fontweight='bold', fontsize=13)

    plt.tight_layout()
    plt.savefig('confusion_matrix_svm_linear.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Karmaşıklık matrisi görseli 'confusion_matrix_svm_linear.png' olarak kaydedildi.")

    return {
        'name': 'SVM (Linear)',
        'accuracy': acc_svm_linear,
        'precision': precision_score(y_test, y_pred_svm_linear),
        'recall': recall_score(y_test, y_pred_svm_linear),
        'f1': f1_score(y_test, y_pred_svm_linear),
        'y_pred_proba': svm_linear.predict_proba(X_test)[:, 1]
    }

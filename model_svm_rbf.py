import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

def run_model(X_train, X_test, y_train, y_test):
    print("\n--- SVM (Kernel: RBF) ---")
    svm_rbf = SVC(kernel='rbf', random_state=42, probability=True)
    svm_rbf.fit(X_train, y_train)
    y_pred_svm_rbf = svm_rbf.predict(X_test)

    acc_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
    print(f"SVM (RBF) Accuracy: {acc_svm_rbf:.4f} ({acc_svm_rbf*100:.2f}%)")
    print("\nClassification Report (RBF):")
    print(classification_report(y_test, y_pred_svm_rbf, target_names=['Gerçek (0)', 'Sahte (1)']))

    cm_svm_rbf = confusion_matrix(y_test, y_pred_svm_rbf)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_svm_rbf, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Tahmin: 0', 'Tahmin: 1'],
                yticklabels=['Asıl: 0', 'Asıl: 1'], annot_kws={"size": 14})
    plt.title(f'SVM (RBF)\nDoğruluk: {acc_svm_rbf*100:.2f}%', fontweight='bold', fontsize=13)

    plt.tight_layout()
    plt.savefig('confusion_matrix_svm_rbf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Karmaşıklık matrisi görseli 'confusion_matrix_svm_rbf.png' olarak kaydedildi.")

    return {
        'name': 'SVM (RBF)',
        'accuracy': acc_svm_rbf,
        'precision': precision_score(y_test, y_pred_svm_rbf),
        'recall': recall_score(y_test, y_pred_svm_rbf),
        'f1': f1_score(y_test, y_pred_svm_rbf),
        'y_pred_proba': svm_rbf.predict_proba(X_test)[:, 1]
    }

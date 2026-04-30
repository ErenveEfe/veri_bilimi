import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

def run_model(X_train, X_test, y_train, y_test):
    print("\n--- Gaussian Naive Bayes (GaussianNB) ---")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)

    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    print(f"GaussianNB Accuracy (Doğruluk): {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")

    print("\nGaussianNB Sınıflandırma Raporu (Classification Report):")
    print(classification_report(y_test, y_pred_nb, target_names=['Gerçek (0)', 'Sahte (1)']))

    cm_nb = confusion_matrix(y_test, y_pred_nb)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tahmin: Gerçek (0)', 'Tahmin: Sahte (1)'],
                yticklabels=['Asıl: Gerçek (0)', 'Asıl: Sahte (1)'],
                annot_kws={"size": 14})
    plt.title(f'GaussianNB Karmaşıklık Matrisi\nDoğruluk: {nb_accuracy*100:.2f}%', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.savefig('confusion_matrix_gaussian_nb.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Karmaşıklık matrisi görseli 'confusion_matrix_gaussian_nb.png' olarak kaydedildi.")

    return {
        'name': 'GaussianNB',
        'accuracy': nb_accuracy,
        'precision': precision_score(y_test, y_pred_nb),
        'recall': recall_score(y_test, y_pred_nb),
        'f1': f1_score(y_test, y_pred_nb),
        'y_pred_proba': nb_model.predict_proba(X_test)[:, 1]
    }

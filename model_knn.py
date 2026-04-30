import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

def run_model(X_train, X_test, y_train, y_test):
    print("\n--- K-Nearest Neighbors (KNN, n_neighbors=5) ---")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    print(f"KNN Accuracy (Doğruluk): {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")

    print("\nKNN Sınıflandırma Raporu (Classification Report):")
    print(classification_report(y_test, y_pred_knn, target_names=['Gerçek (0)', 'Sahte (1)']))

    cm_knn = confusion_matrix(y_test, y_pred_knn)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Tahmin: Gerçek (0)', 'Tahmin: Sahte (1)'],
                yticklabels=['Asıl: Gerçek (0)', 'Asıl: Sahte (1)'],
                annot_kws={"size": 14})
    plt.title(f'KNN Karmaşıklık Matrisi\nDoğruluk: {knn_accuracy*100:.2f}%', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.savefig('confusion_matrix_knn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Karmaşıklık matrisi görseli 'confusion_matrix_knn.png' olarak kaydedildi.")

    return {
        'name': 'KNN',
        'accuracy': knn_accuracy,
        'precision': precision_score(y_test, y_pred_knn),
        'recall': recall_score(y_test, y_pred_knn),
        'f1': f1_score(y_test, y_pred_knn),
        'y_pred_proba': knn_model.predict_proba(X_test)[:, 1]
    }

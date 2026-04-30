import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def run_model(X_train, X_test, y_train, y_test):
    print("\n--- YSA Modeli Kuruluyor ---")
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("\n--- YSA Eğitimi Başlıyor (Epochs: 50, Batch Size: 16) ---")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'], label='Eğitim (Train) Loss', color='blue')
    axes[0].plot(history.history['val_loss'], label='Doğrulama (Val) Loss', color='orange')
    axes[0].set_title('Model Loss (Kayıp) Eğrisi', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Epoch (Tur)', fontsize=12)
    axes[0].set_ylabel('Loss Değeri', fontsize=12)
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Eğitim (Train) Accuracy', color='green')
    axes[1].plot(history.history['val_accuracy'], label='Doğrulama (Val) Accuracy', color='red')
    axes[1].set_title('Model Accuracy (Doğruluk) Eğrisi', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Epoch (Tur)', fontsize=12)
    axes[1].set_ylabel('Accuracy Oranı', fontsize=12)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('ysa_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Öğrenme eğrileri 'ysa_learning_curves.png' olarak kaydedildi.")

    print("\n--- Test Seti Üzerinde Tahmin ---")
    y_pred_probs = model.predict(X_test)
    y_pred_classes = (y_pred_probs >= 0.5).astype(int).flatten()

    acc_ysa = accuracy_score(y_test, y_pred_classes)
    print(f"YSA Accuracy (Doğruluk): {acc_ysa:.4f} ({acc_ysa*100:.2f}%)")

    print("\nYSA Sınıflandırma Raporu (Classification Report):")
    print(classification_report(y_test, y_pred_classes, target_names=['Gerçek (0)', 'Sahte (1)']))

    cm_ysa = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_ysa, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tahmin: 0', 'Tahmin: 1'],
                yticklabels=['Asıl: 0', 'Asıl: 1'],
                annot_kws={"size": 15})
    plt.title(f'YSA Karmaşıklık Matrisi\nDoğruluk: {acc_ysa*100:.2f}%', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('ysa_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] YSA Karmaşıklık matrisi 'ysa_confusion_matrix.png' olarak kaydedildi.")

    return {
        'name': 'YSA',
        'accuracy': acc_ysa,
        'precision': precision_score(y_test, y_pred_classes),
        'recall': recall_score(y_test, y_pred_classes),
        'f1': f1_score(y_test, y_pred_classes),
        'y_pred_proba': y_pred_probs.flatten()
    }

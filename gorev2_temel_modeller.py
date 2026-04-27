# =============================================================================
# BANKNOT DOĞRULAMA PROJESİ
# Görev 2: Temel (Baseline) Modellerin Kurulumu ve Değerlendirilmesi
# =============================================================================

import sys
import io

# Windows konsolunda Türkçe karakter sorunu yasamamak icin stdout encoding ayari
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Grafigi pencerede acmak yerine dosyaya kaydetmek icin
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Grafik stilleri
sns.set_style("whitegrid")

# --- 1. Veriyi Hazırlama (Görev 1'den) ---
sutun_isimleri = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df = pd.read_csv('data_banknote_authentication.txt', header=None, names=sutun_isimleri)

X = df.drop('class', axis=1)
y = df['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print("=" * 70)
print("VERİ HAZIRLIĞI TAMAMLANDI")
print("Eğitim seti boyutu :", X_train.shape)
print("Test seti boyutu   :", X_test.shape)
print("=" * 70)

# --- 2. Gaussian Naive Bayes Modeli ---
print("\n--- 1. Gaussian Naive Bayes (GaussianNB) ---")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, y_pred_nb)
print(f"GaussianNB Accuracy (Doğruluk): {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")

print("\nGaussianNB Sınıflandırma Raporu (Classification Report):")
print(classification_report(y_test, y_pred_nb, target_names=['Gerçek (0)', 'Sahte (1)']))

# --- 3. K-Nearest Neighbors (KNN) Modeli ---
print("\n--- 2. K-Nearest Neighbors (KNN, n_neighbors=5) ---")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy (Doğruluk): {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")

print("\nKNN Sınıflandırma Raporu (Classification Report):")
print(classification_report(y_test, y_pred_knn, target_names=['Gerçek (0)', 'Sahte (1)']))


# --- 4. Karmaşıklık Matrislerinin (Confusion Matrix) Görselleştirilmesi ---
cm_nb = confusion_matrix(y_test, y_pred_nb)
cm_knn = confusion_matrix(y_test, y_pred_knn)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# GaussianNB Confusion Matrix
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Tahmin: Gerçek (0)', 'Tahmin: Sahte (1)'],
            yticklabels=['Asıl: Gerçek (0)', 'Asıl: Sahte (1)'],
            annot_kws={"size": 14})
axes[0].set_title(f'GaussianNB Karmaşıklık Matrisi\nDoğruluk: {nb_accuracy*100:.2f}%', fontweight='bold', fontsize=14)

# KNN Confusion Matrix
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Tahmin: Gerçek (0)', 'Tahmin: Sahte (1)'],
            yticklabels=['Asıl: Gerçek (0)', 'Asıl: Sahte (1)'],
            annot_kws={"size": 14})
axes[1].set_title(f'KNN Karmaşıklık Matrisi\nDoğruluk: {knn_accuracy*100:.2f}%', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
print("\n[OK] Karmaşıklık matrisleri görseli 'confusion_matrices.png' olarak kaydedildi.")

print("\n" + "=" * 70)
print("GÖREV 2 TAMAMLANDI")
print("=" * 70)
print("\n[HAZIR] Görev 3 (Gelişmiş Modeller: SVM ve Random Forest) için hazırız! Komutunuzu bekliyorum.")

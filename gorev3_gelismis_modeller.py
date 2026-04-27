# =============================================================================
# BANKNOT DOĞRULAMA PROJESİ
# Görev 3: Gelişmiş Modeller (SVM ve Random Forest)
# =============================================================================

import sys
import io

# Windows konsolunda Türkçe karakter sorunu yaşamamak için stdout encoding ayarı
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Grafik arayüzü bloklanmasını engellemek için
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Grafik stilleri
sns.set_style("whitegrid")

# --- 1. Veriyi Hazırlama (Önceki Görevlerden) ---
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
print("=" * 70)

# --- 2. Support Vector Classification (SVM) - Linear Kernel ---
print("\n--- 1. SVM (Kernel: Linear) ---")
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_svm_linear = svm_linear.predict(X_test)

acc_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
print(f"SVM (Linear) Accuracy: {acc_svm_linear:.4f} ({acc_svm_linear*100:.2f}%)")
print("\nClassification Report (Linear):")
print(classification_report(y_test, y_pred_svm_linear, target_names=['Gerçek (0)', 'Sahte (1)']))

# --- 3. Support Vector Classification (SVM) - RBF Kernel ---
print("\n--- 2. SVM (Kernel: RBF) ---")
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_test)

acc_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
print(f"SVM (RBF) Accuracy: {acc_svm_rbf:.4f} ({acc_svm_rbf*100:.2f}%)")
print("\nClassification Report (RBF):")
print(classification_report(y_test, y_pred_svm_rbf, target_names=['Gerçek (0)', 'Sahte (1)']))

# --- 4. Random Forest Classifier ---
print("\n--- 3. Random Forest (n_estimators=100) ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=['Gerçek (0)', 'Sahte (1)']))

# --- 5. Özellik Önemi (Feature Importance) ---
importances = rf_model.feature_importances_
feature_names = X.columns

# Azalan sırada indeksleri alma
indices = np.argsort(importances)[::-1]

print("\n--- Özellik Önemi (Feature Importances) ---")
for i in range(X.shape[1]):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Özellik Önemi Bar Grafiği
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
plt.title('Random Forest - Özellik Önemi (Feature Importance)', fontweight='bold', fontsize=14)
plt.xlabel('Önem Derecesi (Bilgi Kazancı)', fontsize=12)
plt.ylabel('Özellik (Feature)', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("\n[OK] Özellik önemi grafiği 'feature_importance.png' olarak kaydedildi.")


# --- 6. Karmaşıklık Matrislerinin Görselleştirilmesi ---
cm_svm_linear = confusion_matrix(y_test, y_pred_svm_linear)
cm_svm_rbf = confusion_matrix(y_test, y_pred_svm_rbf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# SVM Linear
sns.heatmap(cm_svm_linear, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=['Tahmin: 0', 'Tahmin: 1'],
            yticklabels=['Asıl: 0', 'Asıl: 1'], annot_kws={"size": 14})
axes[0].set_title(f'SVM (Linear)\nDoğruluk: {acc_svm_linear*100:.2f}%', fontweight='bold', fontsize=13)

# SVM RBF
sns.heatmap(cm_svm_rbf, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=['Tahmin: 0', 'Tahmin: 1'],
            yticklabels=['Asıl: 0', 'Asıl: 1'], annot_kws={"size": 14})
axes[1].set_title(f'SVM (RBF)\nDoğruluk: {acc_svm_rbf*100:.2f}%', fontweight='bold', fontsize=13)

# Random Forest
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds', ax=axes[2],
            xticklabels=['Tahmin: 0', 'Tahmin: 1'],
            yticklabels=['Asıl: 0', 'Asıl: 1'], annot_kws={"size": 14})
axes[2].set_title(f'Random Forest\nDoğruluk: {acc_rf*100:.2f}%', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('advanced_confusion_matrices.png', dpi=150, bbox_inches='tight')
print("[OK] 3 Modelin Karmaşıklık matrisleri 'advanced_confusion_matrices.png' olarak kaydedildi.")

print("\n" + "=" * 70)
print("GÖREV 3 TAMAMLANDI")
print("=" * 70)
print("\n[HAZIR] Görev 4 (Yapay Sinir Ağları - YSA) için hazırız! Komutunuzu bekliyorum.")

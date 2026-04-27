# =============================================================================
# BANKNOT DOĞRULAMA PROJESİ
# Görev 4: Yapay Sinir Ağları (YSA) ile İkili Sınıflandırma
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os
# TensorFlow loglarını azaltmak için (isteğe bağlı)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# --- 2. YSA Modeli Mimarisi (Architecture) ---
print("\n--- YSA Modeli Kuruluyor ---")
model = Sequential()

# Girdi katmanı ve 1. Gizli Katman
model.add(Dense(16, activation='relu', input_shape=(4,)))

# 2. Gizli Katman
model.add(Dense(8, activation='relu'))

# Çıktı Katmanı (İkili sınıflandırma için sigmoid)
model.add(Dense(1, activation='sigmoid'))

model.summary()

# --- 3. Modeli Derleme (Compile) ---
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- 4. Eğitimi Başlatma (Training) ---
print("\n--- YSA Eğitimi Başlıyor (Epochs: 50, Batch Size: 16) ---")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- 5. Öğrenme Eğrileri (Learning Curves) Çizimi ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Kayıp (Loss) Grafiği
axes[0].plot(history.history['loss'], label='Eğitim (Train) Loss', color='blue')
axes[0].plot(history.history['val_loss'], label='Doğrulama (Val) Loss', color='orange')
axes[0].set_title('Model Loss (Kayıp) Eğrisi', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Epoch (Tur)', fontsize=12)
axes[0].set_ylabel('Loss Değeri', fontsize=12)
axes[0].legend()

# Doğruluk (Accuracy) Grafiği
axes[1].plot(history.history['accuracy'], label='Eğitim (Train) Accuracy', color='green')
axes[1].plot(history.history['val_accuracy'], label='Doğrulama (Val) Accuracy', color='red')
axes[1].set_title('Model Accuracy (Doğruluk) Eğrisi', fontweight='bold', fontsize=14)
axes[1].set_xlabel('Epoch (Tur)', fontsize=12)
axes[1].set_ylabel('Accuracy Oranı', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig('ysa_learning_curves.png', dpi=150, bbox_inches='tight')
print("\n[OK] Öğrenme eğrileri 'ysa_learning_curves.png' olarak kaydedildi.")

# --- 6. Değerlendirme ve Eşik (Threshold) Uygulaması ---
print("\n--- Test Seti Üzerinde Tahmin ---")
# Tahminler olasılık (0.0 - 1.0) olarak döner
y_pred_probs = model.predict(X_test)

# 0.5 eşik değeri ile tam sayı sınıflarına çevirme
y_pred_classes = (y_pred_probs >= 0.5).astype(int).flatten()

acc_ysa = accuracy_score(y_test, y_pred_classes)
print(f"YSA Accuracy (Doğruluk): {acc_ysa:.4f} ({acc_ysa*100:.2f}%)")

print("\nYSA Sınıflandırma Raporu (Classification Report):")
print(classification_report(y_test, y_pred_classes, target_names=['Gerçek (0)', 'Sahte (1)']))

# --- 7. Karmaşıklık Matrisi (Confusion Matrix) ---
cm_ysa = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(7, 5))
sns.heatmap(cm_ysa, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tahmin: 0', 'Tahmin: 1'],
            yticklabels=['Asıl: 0', 'Asıl: 1'],
            annot_kws={"size": 15})
plt.title(f'YSA Karmaşıklık Matrisi\nDoğruluk: {acc_ysa*100:.2f}%', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('ysa_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("[OK] YSA Karmaşıklık matrisi 'ysa_confusion_matrix.png' olarak kaydedildi.")

print("\n" + "=" * 70)
print("GÖREV 4 (FİNAL) TAMAMLANDI")
print("=" * 70)

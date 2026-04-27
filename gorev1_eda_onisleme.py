# =============================================================================
# BANKNOT DOĞRULAMA PROJESİ
# Görev 1: Veri Yükleme, Keşifsel Veri Analizi (EDA) ve Ön İşleme
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- 1. Gerekli Kütüphanelerin Import Edilmesi ---
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (grafigi dosyaya kaydeder, pencere acmaz)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Grafik stilini ayarlıyoruz (daha okunabilir grafikler için)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# --- 2. Veriyi Okuma ve Sütun İsimlerini Atama ---
# Dosyada başlık (header) satırı yok, bu yüzden header=None kullanıyoruz
# Sütun isimlerini 'names' parametresiyle manuel olarak atıyoruz
sutun_isimleri = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

df = pd.read_csv('data_banknote_authentication.txt', header=None, names=sutun_isimleri)

print("=" * 70)
print("VERİ SETİ BAŞARIYLA YÜKLENDİ")
print("=" * 70)

# --- 3. Verinin Genel Yapısını İnceleme ---

# 3a. İlk 5 satır
print("\n--- İlk 5 Satır (head) ---")
print(df.head())

# 3b. Son 5 satır
print("\n--- Son 5 Satır (tail) ---")
print(df.tail())

# 3c. Veri seti boyutu
print(f"\n--- Veri Seti Boyutu ---")
print(f"Satır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")

# 3d. Veri tipleri ve genel bilgi
print("\n--- Veri Seti Bilgisi (info) ---")
df.info()

# 3e. Temel istatistiksel özet
print("\n--- İstatistiksel Özet (describe) ---")
print(df.describe().round(4))

# --- 4. Eksik Değer Kontrolü ---
print("\n--- Eksik Değer Kontrolü ---")
eksik_degerler = df.isnull().sum()
print(eksik_degerler)
print(f"\nToplam eksik değer sayısı: {eksik_degerler.sum()}")

if eksik_degerler.sum() == 0:
    print("[OK] Veri setinde HİÇBİR eksik değer bulunmamaktadır.")
else:
    print("[UYARI] Veri setinde eksik değerler mevcut! İlgili işlemler yapılmalıdır.")

# --- 5. Hedef Değişken (class) Dağılımı ---
print("\n--- Hedef Değişken Dağılımı ---")
sinif_dagilimi = df['class'].value_counts()
print(sinif_dagilimi)
print(f"\nSınıf oranları:")
print(df['class'].value_counts(normalize=True).round(4) * 100)

# Hedef değişken dağılım grafiği
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar grafiği
renk_paleti = ['#2ecc71', '#e74c3c']  # Yeşil: Gerçek (0), Kırmızı: Sahte (1)
sinif_etiketleri = ['Gerçek (0)', 'Sahte (1)']

bars = axes[0].bar(sinif_etiketleri, sinif_dagilimi.values, color=renk_paleti,
                   edgecolor='black', linewidth=1.2)

# Bar üzerine değerleri yazma
for bar, val in zip(bars, sinif_dagilimi.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(val), ha='center', va='bottom', fontweight='bold', fontsize=13)

axes[0].set_title('Hedef Değişken (class) Dağılımı', fontweight='bold', fontsize=14)
axes[0].set_ylabel('Gözlem Sayısı')
axes[0].set_xlabel('Sınıf')

# Pasta grafiği
axes[1].pie(sinif_dagilimi.values, labels=sinif_etiketleri, autopct='%1.1f%%',
            colors=renk_paleti, startangle=90, explode=(0.03, 0.03),
            textprops={'fontsize': 12}, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
axes[1].set_title('Sınıf Oranları (%)', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('sinif_dagilimi.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Sınıf dağılım grafiği 'sinif_dagilimi.png' olarak kaydedildi.")

# --- 6. Özelliklerin Ölçeklendirilmesi (StandardScaler) ---
# Bağımsız değişkenler (features) ve hedef değişkeni (target) ayırıyoruz
X = df.drop('class', axis=1)  # Özellikler (variance, skewness, curtosis, entropy)
y = df['class']                # Hedef değişken (0 veya 1)

print("\n--- Ölçeklendirme (StandardScaler) ---")
print(f"Ölçeklendirme ÖNCESİ - Özellik ortalamaları:\n{X.mean().round(4)}")
print(f"\nÖlçeklendirme ÖNCESİ - Özellik standart sapmaları:\n{X.std().round(4)}")

# StandardScaler nesnesi oluşturma ve fit_transform uygulama
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ölçeklenmiş veriyi DataFrame'e çeviriyoruz (okunabilirlik için)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print(f"\nÖlçeklendirme SONRASI - Özellik ortalamaları (~0 olmalı):\n{X_scaled_df.mean().round(4)}")
print(f"\nÖlçeklendirme SONRASI - Özellik standart sapmaları (~1 olmalı):\n{X_scaled_df.std().round(4)}")

# --- 7. Eğitim ve Test Setlerine Ayırma (Train/Test Split) ---
# %80 Eğitim, %20 Test - random_state sabitleyerek tekrarlanabilirlik sağlıyoruz
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print("\n--- Eğitim / Test Ayrımı ---")
print(f"Eğitim seti boyutu : X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test seti boyutu   : X_test={X_test.shape}, y_test={y_test.shape}")

print(f"\nEğitim setinde sınıf dağılımı:")
print(y_train.value_counts())
print(f"\nTest setinde sınıf dağılımı:")
print(y_test.value_counts())

# --- ÖZET ---
print("\n" + "=" * 70)
print("GÖREV 1 TAMAMLANDI - ÖZET")
print("=" * 70)
print(f"• Toplam gözlem sayısı  : {df.shape[0]}")
print(f"• Toplam özellik sayısı : {X.shape[1]}")
print(f"• Eksik değer           : {eksik_degerler.sum()}")
print(f"• Sınıf 0 (Gerçek)     : {sinif_dagilimi[0]} ({sinif_dagilimi[0]/len(df)*100:.1f}%)")
print(f"• Sınıf 1 (Sahte)      : {sinif_dagilimi[1]} ({sinif_dagilimi[1]/len(df)*100:.1f}%)")
print(f"• Eğitim seti           : {X_train.shape[0]} gözlem")
print(f"• Test seti             : {X_test.shape[0]} gözlem")
print(f"• Ölçeklendirme         : StandardScaler uygulandı")
print("=" * 70)
print("\n[HAZIR] Görev 2 için hazırız! Komutunuzu bekliyorum.")

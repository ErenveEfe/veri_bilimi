import os
import sys
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import model_gaussian_nb
import model_knn
import model_svm_linear
import model_svm_rbf
import model_random_forest
try:
    import model_ysa
    ysa_available = True
except ImportError:
    print("UYARI: TensorFlow/Keras kütüphaneleri bulunamadığı için YSA modeli atlanacak.")
    ysa_available = False

def load_and_preprocess_data(dataset_path):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

    sutun_isimleri = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    df = pd.read_csv(dataset_path, header=None, names=sutun_isimleri)

    print("VERI SETI BASARIYLA YUKLENDI")

    # İlk 5 satır
    print("\n--- ILK 5 SATIR ---")
    print(df.head())

    # Son 5 satır
    print("\n--- SON 5 SATIR ---")
    print(df.tail())

    # Veri seti boyutu
    print(f"\n--- VERI SETI BOYUTU ---")
    print(f"SATIR SAYISI: {df.shape[0]}, SUTUN SAYISI: {df.shape[1]}")

    # Veri tipleri ve genel bilgi
    print("\n--- VERI SETI BILGISI ---")
    df.info()

    # Temel istatistiksel özet
    print("\n--- ISTATISTIKSEL OZET ---")
    print(df.describe().round(4))

    # Eksik Değer Kontrolü
    print("\n--- EKSIK DEGER KONTROLU ---")
    eksik_degerler = df.isnull().sum()
    print(eksik_degerler)
    print(f"\nTOPLAM EKSIK DEGER SAYISI: {eksik_degerler.sum()}")

    if eksik_degerler.sum() == 0:
        print("VERI SETINDE HICBIR EKSIK DEGER BULUNAMAMISTIR.")
    else:
        print("VERI SETINDE EKSIK DEGERLER MEVCUTTUR! ILGILI ISLEMLER YAPILMALIDIR.")

    # Hedef Değişken (class) Dağılımı
    print("\n--- HEDEF DEGISKEN DAGILIMI ---")
    sinif_dagilimi = df['class'].value_counts()
    print(sinif_dagilimi)
    print(f"\nSINIF ORANLARI:")
    print(df['class'].value_counts(normalize=True).round(4) * 100)

    # Hedef değişken dağılım grafiği
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    renk_paleti = ['#2ecc71', '#e74c3c']
    sinif_etiketleri = ['GERCEK (0)', 'SAHTE (1)']

    bars = axes[0].bar(sinif_etiketleri, sinif_dagilimi.values, color=renk_paleti,
                       edgecolor='black', linewidth=1.2)

    for bar, val in zip(bars, sinif_dagilimi.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     str(val), ha='center', va='bottom', fontweight='bold', fontsize=13)

    axes[0].set_title('Hedef Değişken (class) Dağılımı', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('Gözlem Sayısı')
    axes[0].set_xlabel('Sınıf')

    axes[1].pie(sinif_dagilimi.values, labels=sinif_etiketleri, autopct='%1.1f%%',
                colors=renk_paleti, startangle=90, explode=(0.03, 0.03),
                textprops={'fontsize': 12}, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    axes[1].set_title('Sınıf Oranları (%)', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.savefig('sinif_dagilimi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Sınıf dağılım grafiği 'sinif_dagilimi.png' olarak kaydedildi.")

    # ------------------ YENİ EDA GRAFİKLERİ ------------------
    # 1. Korelasyon Matrisi (Heatmap)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Korelasyon Matrisi", fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('korelasyon_matrisi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Korelasyon matrisi 'korelasyon_matrisi.png' olarak kaydedildi.")

    # 2. Çiftli Dağılım Grafiği (Pairplot)
    g = sns.pairplot(df, hue='class', palette=renk_paleti, markers=["o", "s"], corner=True)
    g.fig.suptitle("Değişkenlerin İkili Dağılımı (Pairplot)", y=1.02, fontweight='bold', fontsize=14)
    plt.savefig('ikili_dagilim_pairplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] İkili dağılım grafiği 'ikili_dagilim_pairplot.png' olarak kaydedildi.")

    # 3. Kutu Grafiği (Boxplot)
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(df.columns[:-1]):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='class', y=col, data=df, palette=renk_paleti)
        plt.title(f'{col} Dağılımı (Boxplot)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('kutu_grafikleri_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Kutu grafikleri 'kutu_grafikleri_boxplot.png' olarak kaydedildi.")

    # 4. Yoğunluk Grafikleri (KDE)
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(df.columns[:-1]):
        plt.subplot(2, 2, i+1)
        sns.kdeplot(data=df, x=col, hue='class', fill=True, palette=renk_paleti, alpha=0.5)
        plt.title(f'{col} Yoğunluk Dağılımı (KDE)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('yogunluk_grafikleri_kde.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Yoğunluk grafikleri 'yogunluk_grafikleri_kde.png' olarak kaydedildi.")
    # ---------------------------------------------------------

    # Özelliklerin Ölçeklendirilmesi (StandardScaler)
    X = df.drop('class', axis=1)
    y = df['class']

    print("\n--- Ölçeklendirme (StandardScaler) ---")
    print(f"Ölçeklendirme ÖNCESİ - Özellik ortalamaları:\n{X.mean().round(4)}")
    print(f"\nÖlçeklendirme ÖNCESİ - Özellik standart sapmaları:\n{X.std().round(4)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"\nÖlçeklendirme SONRASI - Özellik ortalamaları (~0 olmalı):\n{X_scaled_df.mean().round(4)}")
    print(f"\nÖlçeklendirme SONRASI - Özellik standart sapmaları (~1 olmalı):\n{X_scaled_df.std().round(4)}")

    # ------------------ PCA İLE 2 BOYUTLU GÖRSELLEŞTİRME ------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=renk_paleti, alpha=0.7)
    plt.title(f"PCA - 2 Boyutlu Dağılım (Varyans Açıklama: %{sum(pca.explained_variance_ratio_)*100:.1f})", fontweight='bold')
    plt.xlabel("1. Temel Bileşen (PC1)")
    plt.ylabel("2. Temel Bileşen (PC2)")
    plt.tight_layout()
    plt.savefig('pca_2b_dagilim.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] PCA 2 boyutlu dağılım grafiği 'pca_2b_dagilim.png' olarak kaydedildi.")
    # ----------------------------------------------------------------------

    # Eğitim ve Test Setlerine Ayırma (Train/Test Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n--- Eğitim / Test Ayrımı ---")
    print(f"Eğitim seti boyutu : X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test seti boyutu   : X_test={X_test.shape}, y_test={y_test.shape}")

    print("\n" + "=" * 70)
    print("İŞLEM TAMAMLANDI - ÖZET")
    print("=" * 70)
    print(f"• Toplam gözlem sayısı  : {df.shape[0]}")
    print(f"• Toplam özellik sayısı : {X.shape[1]}")
    print(f"• Eğitim seti           : {X_train.shape[0]} gözlem")
    print(f"• Test seti             : {X_test.shape[0]} gözlem")
    print("=" * 70)
    
    return X_train, X_test, y_train, y_test, X.columns

def main():
    # Veri setinin ismini buradan değiştirebilirsiniz
    DATASET_PATH = 'data_banknote_authentication_10k.txt'
    
    print("Proje Başlatılıyor...\n")
    
    # Scriptin bulunduğu klasörün yolunu alalım (veri_bilimi klasörü)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Veri seti yolu
    abs_dataset_path = os.path.join(script_dir, DATASET_PATH)
    
    # Çıktı klasörünü de scriptin yanına oluşturalım
    dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
    output_dir = os.path.join(script_dir, dataset_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    print(f"Çıktılar '{output_dir}' klasörüne kaydedilecek.\n")
    
    try:
        # 1. Veri Yükleme ve Ön İşleme
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(abs_dataset_path)
        
        # 2. Modelleri Çalıştırma
        print("\n" + "="*70)
        print("MODELLER ÇALIŞTIRILIYOR...")
        print("="*70)
        
        results = []
        results.append(model_gaussian_nb.run_model(X_train, X_test, y_train, y_test))
        results.append(model_knn.run_model(X_train, X_test, y_train, y_test))
        results.append(model_svm_linear.run_model(X_train, X_test, y_train, y_test))
        results.append(model_svm_rbf.run_model(X_train, X_test, y_train, y_test))
        results.append(model_random_forest.run_model(X_train, X_test, y_train, y_test, feature_names))
        if ysa_available:
            results.append(model_ysa.run_model(X_train, X_test, y_train, y_test))
        
        print("\n" + "="*70)
        print("MODEL PERFORMANS KARŞILAŞTIRMA GÖRSELLERİ OLUŞTURULUYOR...")
        print("="*70)

        # 1. Bar Chart (Modeller Arası Başarı Kıyaslama)
        df_results = pd.DataFrame(results)
        
        plt.figure(figsize=(12, 6))
        df_melted = df_results[['name', 'accuracy', 'precision', 'recall', 'f1']].melt(id_vars='name', var_name='Metric', value_name='Score')
        sns.barplot(x='name', y='Score', hue='Metric', data=df_melted, palette='viridis')
        plt.title('Modeller Arası Başarı Kıyaslama Tablosu', fontweight='bold', fontsize=14)
        plt.ylim(0, 1.1)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('model_karsilastirma_bar_chart.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[OK] Modeller arası başarı kıyaslama grafiği 'model_karsilastirma_bar_chart.png' olarak kaydedildi.")

        # 2. ROC Curve
        plt.figure(figsize=(10, 8))
        for res in results:
            fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{res['name']} (AUC = {roc_auc:.3f})")
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
        plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
        plt.title('Modeller İçin ROC Eğrisi Karşılaştırması', fontweight='bold', fontsize=14)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_egrisi_karsilastirma.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[OK] ROC Eğrisi karşılaştırması 'roc_egrisi_karsilastirma.png' olarak kaydedildi.")

        # 3. Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        for res in results:
            precision, recall, _ = precision_recall_curve(y_test, res['y_pred_proba'])
            plt.plot(recall, precision, lw=2, label=f"{res['name']}")
            
        plt.xlabel('Recall (Duyarlılık)')
        plt.ylabel('Precision (Hassasiyet)')
        plt.title('Modeller İçin Precision-Recall Eğrisi Karşılaştırması', fontweight='bold', fontsize=14)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig('precision_recall_egrisi_karsilastirma.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[OK] Precision-Recall Eğrisi karşılaştırması 'precision_recall_egrisi_karsilastirma.png' olarak kaydedildi.")
        
        print("\n" + "="*70)
        print("TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
        print("="*70)
        
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()

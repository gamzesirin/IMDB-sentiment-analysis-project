import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'


class IMDBDataLoader:
    """
    IMDB veri setini yükleyen ve keşfeden sınıf.
    
    Attributes:
        data_path (str): Veri dosyasının yolu
        df (pd.DataFrame): Yüklenen veri çerçevesi
    """
    
    def __init__(self, data_path: str = None):
        """
        DataLoader sınıfını başlatır.
        
        Args:
            data_path: CSV dosyasının yolu
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        CSV dosyasından veriyi yükler.
        
        Args:
            data_path: CSV dosyasının yolu
            
        Returns:
            pd.DataFrame: Yüklenen veri çerçevesi
        """
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("Veri yolu belirtilmedi!")
            
        print(f"Veri yukleniyor: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"Veri basariyla yuklendi!")
        print(f"   Toplam ornek sayisi: {len(self.df):,}")
        
        return self.df
    
    def get_basic_info(self) -> dict:
        """
        Veri seti hakkında temel bilgileri döndürür.
        
        Returns:
            dict: Temel istatistikler
        """
        if self.df is None:
            raise ValueError("Önce veri yüklenmeli!")
            
        info = {
            'toplam_ornek': len(self.df),
            'sutunlar': list(self.df.columns),
            'veri_tipleri': self.df.dtypes.to_dict(),
            'eksik_degerler': self.df.isnull().sum().to_dict(),
            'bellek_kullanimi': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        return info
    
    def print_data_summary(self):
        """Veri seti özetini yazdırır."""
        if self.df is None:
            raise ValueError("Önce veri yüklenmeli!")
            
        print("\n" + "="*60)
        print("VERI SETI OZETI")
        print("="*60)

        print(f"\nBoyut: {self.df.shape[0]:,} satir x {self.df.shape[1]} sutun")
        print(f"\nSutunlar: {list(self.df.columns)}")

        print(f"\nVeri Tipleri:")
        for col, dtype in self.df.dtypes.items():
            print(f"   - {col}: {dtype}")

        print(f"\nEksik Degerler:")
        for col, missing in self.df.isnull().sum().items():
            print(f"   - {col}: {missing} ({missing/len(self.df)*100:.2f}%)")

        print(f"\nEtiket Dagilimi:")
        if 'sentiment' in self.df.columns:
            label_counts = self.df['sentiment'].value_counts()
            for label, count in label_counts.items():
                print(f"   - {label}: {count:,} ({count/len(self.df)*100:.1f}%)")

        print("\nIlk 3 Ornek:")
        for idx, row in self.df.head(3).iterrows():
            review = row['review'][:100] + "..." if len(row['review']) > 100 else row['review']
            print(f"   [{idx}] Sentiment: {row['sentiment']}")
            print(f"       Review: {review}\n")
            
    def analyze_text_lengths(self) -> pd.DataFrame:
        """
        Metin uzunluklarını analiz eder.
        
        Returns:
            pd.DataFrame: Uzunluk istatistikleri
        """
        if self.df is None:
            raise ValueError("Önce veri yüklenmeli!")
            
        # Karakter ve kelime sayıları
        self.df['char_count'] = self.df['review'].apply(len)
        self.df['word_count'] = self.df['review'].apply(lambda x: len(x.split()))
        
        stats = {
            'Metrik': ['Karakter Sayısı', 'Kelime Sayısı'],
            'Ortalama': [
                self.df['char_count'].mean(),
                self.df['word_count'].mean()
            ],
            'Std': [
                self.df['char_count'].std(),
                self.df['word_count'].std()
            ],
            'Min': [
                self.df['char_count'].min(),
                self.df['word_count'].min()
            ],
            'Max': [
                self.df['char_count'].max(),
                self.df['word_count'].max()
            ],
            'Medyan': [
                self.df['char_count'].median(),
                self.df['word_count'].median()
            ]
        }
        
        return pd.DataFrame(stats)
    
    def plot_sentiment_distribution(self, save_path: str = None):
        """
        Duygu dağılımını görselleştirir.
        
        Args:
            save_path: Grafik kaydetme yolu
        """
        if self.df is None:
            raise ValueError("Önce veri yüklenmeli!")
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        colors = ['#e74c3c', '#27ae60']  # Kırmızı: negative, Yeşil: positive
        sentiment_counts = self.df['sentiment'].value_counts()
        
        ax1 = axes[0]
        bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        ax1.set_title('Duygu Sınıflarının Dağılımı', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Duygu Sınıfı', fontsize=12)
        ax1.set_ylabel('Örnek Sayısı', fontsize=12)
        
        # Bar üzerine değer yazma
        for bar, count in zip(bars, sentiment_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'{count:,}', ha='center', fontsize=11, fontweight='bold')
        
        # Pie chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(
            sentiment_counts.values, 
            labels=sentiment_counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.02, 0.02)
        )
        ax2.set_title('Duygu Sınıflarının Oransal Dağılımı', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {save_path}")

        plt.show()

    def plot_text_length_distribution(self, save_path: str = None):
        """
        Metin uzunluk dağılımlarını görselleştirir.
        
        Args:
            save_path: Grafik kaydetme yolu
        """
        if self.df is None:
            raise ValueError("Önce veri yüklenmeli!")
            
        # Uzunluk hesaplamaları
        if 'word_count' not in self.df.columns:
            self.analyze_text_lengths()
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Kelime sayısı histogramı
        ax1 = axes[0, 0]
        ax1.hist(self.df['word_count'], bins=50, color='#3498db', alpha=0.7, edgecolor='white')
        ax1.axvline(self.df['word_count'].mean(), color='red', linestyle='--', 
                   label=f'Ortalama: {self.df["word_count"].mean():.0f}')
        ax1.axvline(self.df['word_count'].median(), color='green', linestyle='--',
                   label=f'Medyan: {self.df["word_count"].median():.0f}')
        ax1.set_title('Kelime Sayısı Dağılımı', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Kelime Sayısı')
        ax1.set_ylabel('Frekans')
        ax1.legend()
        
        # 2. Sınıflara göre kelime sayısı
        ax2 = axes[0, 1]
        colors = {'negative': '#e74c3c', 'positive': '#27ae60'}
        for sentiment in self.df['sentiment'].unique():
            data = self.df[self.df['sentiment'] == sentiment]['word_count']
            ax2.hist(data, bins=50, alpha=0.5, label=sentiment, color=colors.get(sentiment, 'gray'))
        ax2.set_title('Sınıflara Göre Kelime Sayısı Dağılımı', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Kelime Sayısı')
        ax2.set_ylabel('Frekans')
        ax2.legend()
        
        # 3. Box plot - Sınıflara göre kelime sayısı
        ax3 = axes[1, 0]
        self.df.boxplot(column='word_count', by='sentiment', ax=ax3)
        ax3.set_title('Sınıflara Göre Kelime Sayısı (Box Plot)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Duygu Sınıfı')
        ax3.set_ylabel('Kelime Sayısı')
        plt.suptitle('')  # Otomatik başlığı kaldır
        
        # 4. Violin plot
        ax4 = axes[1, 1]
        parts = ax4.violinplot(
            [self.df[self.df['sentiment'] == 'negative']['word_count'],
             self.df[self.df['sentiment'] == 'positive']['word_count']],
            positions=[1, 2],
            showmeans=True,
            showmedians=True
        )
        ax4.set_xticks([1, 2])
        ax4.set_xticklabels(['Negative', 'Positive'])
        ax4.set_title('Sınıflara Göre Kelime Sayısı (Violin Plot)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Duygu Sınıfı')
        ax4.set_ylabel('Kelime Sayısı')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {save_path}")

        plt.show()

    def get_sample_reviews(self, n: int = 5, sentiment: str = None) -> pd.DataFrame:
        """
        Örnek yorumları döndürür.
        
        Args:
            n: Örnek sayısı
            sentiment: Filtrelenecek duygu sınıfı ('positive' veya 'negative')
            
        Returns:
            pd.DataFrame: Örnek yorumlar
        """
        if self.df is None:
            raise ValueError("Önce veri yüklenmeli!")
            
        if sentiment:
            samples = self.df[self.df['sentiment'] == sentiment].sample(n)
        else:
            samples = self.df.sample(n)
            
        return samples[['review', 'sentiment']]


# Test kodu
if __name__ == "__main__":
    # Örnek kullanım
    loader = IMDBDataLoader()
    
    # Test için küçük bir veri oluştur
    test_data = pd.DataFrame({
        'review': [
            'This movie was great! I loved it.',
            'Terrible film. Waste of time.',
            'Amazing storyline and great acting.',
            'Boring and predictable. Not recommended.',
            'One of the best movies I have ever seen!'
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
    })
    
    # Test dosyası kaydet
    test_path = 'test_data.csv'
    test_data.to_csv(test_path, index=False)
    
    # Test et
    loader.load_data(test_path)
    loader.print_data_summary()
    
    # Temizlik
    os.remove(test_path)
    print("\nTest basariyla tamamlandi!")

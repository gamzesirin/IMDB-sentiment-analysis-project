import re
import string
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from collections import Counter
import pickle
import os

# TensorFlow/Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sklearn
from sklearn.model_selection import train_test_split

# Progress bar
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# Varsayılan İngilizce stop words listesi (NLTK olmadan)
DEFAULT_STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
}


class TextPreprocessor:
    """
    Metin ön işleme sınıfı.
    
    Bu sınıf, ham metin verilerini derin öğrenme modelleri için
    uygun formata dönüştürür.
    
    Attributes:
        max_words (int): Kelime haznesindeki maksimum kelime sayısı
        max_len (int): Maksimum dizi uzunluğu
        tokenizer (Tokenizer): Keras tokenizer nesnesi
        stop_words (set): İngilizce stop words seti
    """
    
    def __init__(self, max_words: int = 10000, max_len: int = 200):
        """
        TextPreprocessor sınıfını başlatır.
        
        Args:
            max_words: Kelime haznesindeki maksimum kelime sayısı
            max_len: Maksimum dizi uzunluğu
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None
        self.stop_words = DEFAULT_STOP_WORDS
        
        # İşlem istatistikleri
        self.stats = {
            'original_vocab_size': 0,
            'final_vocab_size': 0,
            'avg_sequence_length': 0,
            'truncated_sequences': 0
        }
        
    def clean_html(self, text: str) -> str:
        """
        HTML etiketlerini temizler.
        
        Args:
            text: Temizlenecek metin
            
        Returns:
            str: Temizlenmiş metin
        """
        # HTML etiketlerini kaldır
        clean = re.compile('<.*?>')
        text = re.sub(clean, ' ', text)
        
        # HTML özel karakterlerini dönüştür
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        return text
    
    def remove_special_chars(self, text: str, keep_punctuation: bool = False) -> str:
        """
        Özel karakterleri kaldırır.
        
        Args:
            text: İşlenecek metin
            keep_punctuation: Noktalama işaretlerini koru
            
        Returns:
            str: İşlenmiş metin
        """
        # URL'leri kaldır
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Email adreslerini kaldır
        text = re.sub(r'\S+@\S+', '', text)
        
        # Sayıları kaldır (opsiyonel)
        text = re.sub(r'\d+', '', text)
        
        if not keep_punctuation:
            # Noktalama işaretlerini kaldır
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Fazla boşlukları temizle
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Stop words'leri kaldırır.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Stop words'ler kaldırılmış metin
        """
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Metni lemmatize eder (basitleştirilmiş versiyon).
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: İşlenmiş metin (basit suffix kaldırma)
        """
        # Basit lemmatization - sadece yaygın suffix'leri kaldır
        words = text.split()
        lemmatized = []
        for word in words:
            # Basit suffix kaldırma
            if word.endswith('ing') and len(word) > 5:
                word = word[:-3]
            elif word.endswith('ed') and len(word) > 4:
                word = word[:-2]
            elif word.endswith('ly') and len(word) > 4:
                word = word[:-2]
            lemmatized.append(word)
        return ' '.join(lemmatized)
    
    def stem_text(self, text: str) -> str:
        """
        Metni stem eder (basitleştirilmiş versiyon).
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Stem edilmiş metin
        """
        # Basit stemming
        words = text.split()
        stemmed = []
        for word in words:
            # Daha agresif suffix kaldırma
            if word.endswith('tion') and len(word) > 6:
                word = word[:-4]
            elif word.endswith('ness') and len(word) > 6:
                word = word[:-4]
            elif word.endswith('ing') and len(word) > 5:
                word = word[:-3]
            elif word.endswith('ed') and len(word) > 4:
                word = word[:-2]
            elif word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
                word = word[:-1]
            stemmed.append(word)
        return ' '.join(stemmed)
    
    def preprocess_text(self, text: str, 
                       remove_html: bool = True,
                       remove_special: bool = True,
                       lowercase: bool = True,
                       remove_stop: bool = False,
                       lemmatize: bool = False,
                       stem: bool = False) -> str:
        """
        Tek bir metin için tüm ön işleme adımlarını uygular.
        
        Args:
            text: İşlenecek metin
            remove_html: HTML etiketlerini kaldır
            remove_special: Özel karakterleri kaldır
            lowercase: Küçük harfe dönüştür
            remove_stop: Stop words kaldır
            lemmatize: Lemmatization uygula
            stem: Stemming uygula
            
        Returns:
            str: İşlenmiş metin
        """
        if pd.isna(text):
            return ""
            
        # Adım 1: HTML temizleme
        if remove_html:
            text = self.clean_html(text)
        
        # Adım 2: Küçük harfe dönüştürme
        if lowercase:
            text = text.lower()
        
        # Adım 3: Özel karakterleri kaldırma
        if remove_special:
            text = self.remove_special_chars(text)
        
        # Adım 4: Stop words kaldırma (opsiyonel - genelde derin öğrenmede kaldırılmaz)
        if remove_stop:
            text = self.remove_stopwords(text)
        
        # Adım 5: Lemmatization (opsiyonel)
        if lemmatize:
            text = self.lemmatize_text(text)
        
        # Adım 6: Stemming (opsiyonel - lemmatization ile birlikte kullanılmaz)
        if stem and not lemmatize:
            text = self.stem_text(text)
        
        return text
    
    def preprocess_corpus(self, texts: List[str], 
                         show_progress: bool = True,
                         **kwargs) -> List[str]:
        """
        Tüm metin koleksiyonunu ön işler.
        
        Args:
            texts: Metin listesi
            show_progress: İlerleme çubuğu göster
            **kwargs: preprocess_text fonksiyonuna geçirilecek parametreler
            
        Returns:
            List[str]: İşlenmiş metinler
        """
        print(" Metin ön işleme başlıyor...")
        
        processed = []
        iterator = tqdm(texts, desc="Ön işleme") if show_progress else texts
        
        for text in iterator:
            processed.append(self.preprocess_text(text, **kwargs))
        
        print(f" {len(processed):,} metin başarıyla ön işlendi!")
        return processed
    
    def fit_tokenizer(self, texts: List[str]):
        """
        Tokenizer'ı metinlere göre eğitir.
        
        Args:
            texts: Eğitim metinleri
        """
        print(f" Tokenizer eğitiliyor (max_words={self.max_words:,})...")
        
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        self.stats['original_vocab_size'] = len(self.tokenizer.word_index)
        self.stats['final_vocab_size'] = min(self.max_words, len(self.tokenizer.word_index))
        
        print(f" Tokenizer eğitildi!")
        print(f"   Orijinal kelime haznesi: {self.stats['original_vocab_size']:,}")
        print(f"   Kullanılan kelime haznesi: {self.stats['final_vocab_size']:,}")
    
    def texts_to_sequences(self, texts: List[str], 
                          padding: str = 'post',
                          truncating: str = 'post') -> np.ndarray:
        """
        Metinleri sayısal dizilere dönüştürür.
        
        Args:
            texts: Metin listesi
            padding: Padding yönü ('pre' veya 'post')
            truncating: Truncation yönü ('pre' veya 'post')
            
        Returns:
            np.ndarray: Padding uygulanmış sayısal diziler
        """
        if self.tokenizer is None:
            raise ValueError("Önce tokenizer eğitilmeli (fit_tokenizer)!")
        
        print(f" Metinler dizilere dönüştürülüyor (max_len={self.max_len})...")
        
        # Metinleri dizilere dönüştür
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Truncation istatistiği
        self.stats['truncated_sequences'] = sum(1 for seq in sequences if len(seq) > self.max_len)
        
        # Padding uygula
        padded = pad_sequences(sequences, 
                              maxlen=self.max_len, 
                              padding=padding,
                              truncating=truncating)
        
        # Ortalama dizi uzunluğu
        orig_lengths = [len(seq) for seq in sequences]
        self.stats['avg_sequence_length'] = np.mean(orig_lengths)
        
        print(f" Dönüştürme tamamlandı!")
        print(f"   Ortalama dizi uzunluğu: {self.stats['avg_sequence_length']:.1f}")
        print(f"   Kırpılan dizi sayısı: {self.stats['truncated_sequences']:,}")
        print(f"   Çıktı şekli: {padded.shape}")
        
        return padded
    
    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """
        Etiketleri sayısal değerlere dönüştürür.
        
        Args:
            labels: Etiket listesi ('positive' veya 'negative')
            
        Returns:
            np.ndarray: Sayısal etiketler (0 veya 1)
        """
        label_map = {'negative': 0, 'positive': 1}
        encoded = np.array([label_map.get(label, 0) for label in labels])
        
        print(f" Etiketler kodlandı: negative=0, positive=1")
        print(f"   Negatif sayısı: {np.sum(encoded == 0):,}")
        print(f"   Pozitif sayısı: {np.sum(encoded == 1):,}")
        
        return encoded
    
    def prepare_data(self, df: pd.DataFrame,
                    text_column: str = 'review',
                    label_column: str = 'sentiment',
                    test_size: float = 0.2,
                    val_size: float = 0.1,
                    random_state: int = 42,
                    **preprocess_kwargs) -> Tuple:
        """
        Veriyi eğitim, doğrulama ve test setlerine ayırır ve hazırlar.
        
        Args:
            df: Veri çerçevesi
            text_column: Metin sütunu adı
            label_column: Etiket sütunu adı
            test_size: Test seti oranı
            val_size: Doğrulama seti oranı
            random_state: Rastgelelik için seed değeri
            **preprocess_kwargs: Ön işleme parametreleri
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n" + "="*60)
        print(" VERİ HAZIRLAMA")
        print("="*60)
        
        # 1. Metin ön işleme
        texts = self.preprocess_corpus(df[text_column].tolist(), **preprocess_kwargs)
        
        # 2. Etiket kodlama
        labels = self.encode_labels(df[label_column].tolist())
        
        # 3. Train-Test ayrımı
        print(f"\n Veri bölünüyor (test={test_size}, val={val_size})...")
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Validation set
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        print(f"   Eğitim seti: {len(X_train):,}")
        print(f"   Doğrulama seti: {len(X_val):,}")
        print(f"   Test seti: {len(X_test):,}")
        
        # 4. Tokenizer eğitimi (sadece eğitim verisi ile)
        self.fit_tokenizer(X_train)
        
        # 5. Metinleri dizilere dönüştür
        print("\n Eğitim verisi dönüştürülüyor...")
        X_train_seq = self.texts_to_sequences(X_train)
        
        print("\n Doğrulama verisi dönüştürülüyor...")
        X_val_seq = self.texts_to_sequences(X_val)
        
        print("\n Test verisi dönüştürülüyor...")
        X_test_seq = self.texts_to_sequences(X_test)
        
        print("\n" + "="*60)
        print(" VERİ HAZIRLAMA TAMAMLANDI!")
        print("="*60)
        print(f"   X_train shape: {X_train_seq.shape}")
        print(f"   X_val shape: {X_val_seq.shape}")
        print(f"   X_test shape: {X_test_seq.shape}")
        
        return X_train_seq, X_val_seq, X_test_seq, y_train, y_val, y_test
    
    def get_word_index(self) -> dict:
        """
        Kelime-indeks sözlüğünü döndürür.
        
        Returns:
            dict: Kelime-indeks eşleşmeleri
        """
        if self.tokenizer is None:
            raise ValueError("Önce tokenizer eğitilmeli!")
        return self.tokenizer.word_index
    
    def get_vocabulary_size(self) -> int:
        """
        Kelime haznesi boyutunu döndürür.
        
        Returns:
            int: Kelime haznesi boyutu
        """
        if self.tokenizer is None:
            raise ValueError("Önce tokenizer eğitilmeli!")
        return min(self.max_words, len(self.tokenizer.word_index)) + 1  # +1 for padding
    
    def save_tokenizer(self, path: str):
        """
        Tokenizer'ı dosyaya kaydeder.
        
        Args:
            path: Kayıt yolu
        """
        if self.tokenizer is None:
            raise ValueError("Önce tokenizer eğitilmeli!")
            
        with open(path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f" Tokenizer kaydedildi: {path}")
    
    def load_tokenizer(self, path: str):
        """
        Tokenizer'ı dosyadan yükler.
        
        Args:
            path: Dosya yolu
        """
        with open(path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f" Tokenizer yüklendi: {path}")
    
    def transform_single_text(self, text: str, **preprocess_kwargs) -> np.ndarray:
        """
        Tek bir metni model için hazırlar.
        
        Args:
            text: İşlenecek metin
            **preprocess_kwargs: Ön işleme parametreleri
            
        Returns:
            np.ndarray: Model için hazır dizi
        """
        if self.tokenizer is None:
            raise ValueError("Önce tokenizer eğitilmeli!")
        
        # Ön işle
        processed = self.preprocess_text(text, **preprocess_kwargs)
        
        # Diziye dönüştür
        sequence = self.tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        
        return padded
    
    def get_stats(self) -> dict:
        """
        İşlem istatistiklerini döndürür.
        
        Returns:
            dict: İstatistikler
        """
        return self.stats


# Test kodu
if __name__ == "__main__":
    # Test
    preprocessor = TextPreprocessor(max_words=5000, max_len=100)
    
    # Örnek metin
    sample_text = "<br>This is a GREAT movie! I loved it so much... 10/10 would recommend!"
    
    print("Orijinal metin:")
    print(sample_text)
    print("\nİşlenmiş metin:")
    print(preprocessor.preprocess_text(sample_text))
    
    print("\n Preprocessor testi başarılı!")

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow uyarılarını azalt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Proje modüllerini içe aktar
from src.data_loader import IMDBDataLoader
from src.preprocessor import TextPreprocessor
from src.models import SentimentModels
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator, compare_models


class SentimentAnalysisPipeline:
    """
    IMDB Duygu Analizi için tam pipeline.
    
    Bu sınıf, veri yüklemeden model değerlendirmeye kadar
    tüm adımları yönetir.
    """
    
    def __init__(self, config: dict = None):
        """
        Pipeline'ı başlatır.
        
        Args:
            config: Yapılandırma sözlüğü
        """
        # Varsayılan yapılandırma
        self.config = {
            'data_path': 'data/IMDB_Dataset.csv',
            'max_words': 10000,
            'max_len': 200,
            'embedding_dim': 128,
            'test_size': 0.2,
            'val_size': 0.1,
            'batch_size': 64,
            'epochs': 10,
            'model_type': 'bilstm',
            'save_dir': 'models',
            'results_dir': 'results',
            'random_state': 42
        }
        
        # Özel yapılandırmayı güncelle
        if config:
            self.config.update(config)
        
        # Bileşenleri başlat
        self.data_loader = None
        self.preprocessor = None
        self.model_builder = None
        self.trainer = None
        self.evaluator = None
        
        # Veri
        self.df = None
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        
        # Model
        self.model = None
        self.history = None
        
        # Dizinleri oluştur
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['results_dir'], exist_ok=True)
        
        print("="*60)
        print("IMDB DUYGU ANALIZI PROJESI")
        print("="*60)
        print(f"   Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
    
    def load_data(self, data_path: str = None):
        """
        Veriyi yükler ve keşfeder.
        
        Args:
            data_path: Veri dosyası yolu
        """
        print("\n" + "="*60)
        print("ADIM 1: VERI YUKLEME VE KESIF")
        print("="*60)
        
        if data_path:
            self.config['data_path'] = data_path
        
        # Veri yükleyici
        self.data_loader = IMDBDataLoader(self.config['data_path'])
        self.df = self.data_loader.load_data()
        
        # Veri özeti
        self.data_loader.print_data_summary()
        
        # Metin uzunluk analizi
        print("\nMetin Uzunluk Istatistikleri:")
        length_stats = self.data_loader.analyze_text_lengths()
        print(length_stats.to_string(index=False))

        # Görselleştirmeler
        print("\nGorsellestirmeler olusturuluyor...")
        
        # Duygu dağılımı
        self.data_loader.plot_sentiment_distribution(
            save_path=f"{self.config['results_dir']}/sentiment_distribution.png"
        )
        
        # Metin uzunluk dağılımı
        self.data_loader.plot_text_length_distribution(
            save_path=f"{self.config['results_dir']}/text_length_distribution.png"
        )
        
        return self.df
    
    def preprocess_data(self):
        """
        Veriyi ön işler ve eğitim/test setlerine ayırır.
        """
        print("\n" + "="*60)
        print("ADIM 2: VERI ON ISLEME")
        print("="*60)
        
        if self.df is None:
            raise ValueError("Önce veri yüklenmeli!")
        
        # Ön işleyici
        self.preprocessor = TextPreprocessor(
            max_words=self.config['max_words'],
            max_len=self.config['max_len']
        )
        
        # Veriyi hazırla
        (self.X_train, self.X_val, self.X_test, 
         self.y_train, self.y_val, self.y_test) = self.preprocessor.prepare_data(
            self.df,
            text_column='review',
            label_column='sentiment',
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['random_state'],
            remove_html=True,
            remove_special=True,
            lowercase=True,
            remove_stop=False,  # Derin öğrenmede genellikle kaldırılmaz
            lemmatize=False
        )
        
        # Tokenizer'ı kaydet
        tokenizer_path = f"{self.config['save_dir']}/tokenizer.pkl"
        self.preprocessor.save_tokenizer(tokenizer_path)
        
        # İstatistikleri yazdır
        print("\nOn Isleme Istatistikleri:")
        stats = self.preprocessor.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return self.X_train, self.X_val, self.X_test
    
    def build_model(self, model_type: str = None):
        """
        Model oluşturur.
        
        Args:
            model_type: Model tipi
        """
        print("\n" + "="*60)
        print("ADIM 3: MODEL OLUSTURMA")
        print("="*60)
        
        if model_type:
            self.config['model_type'] = model_type
        
        # Kelime haznesi boyutu
        vocab_size = self.preprocessor.get_vocabulary_size()
        
        print(f"\nModel Parametreleri:")
        print(f"   Model Tipi: {self.config['model_type']}")
        print(f"   Kelime Haznesi: {vocab_size:,}")
        print(f"   Embedding Boyutu: {self.config['embedding_dim']}")
        print(f"   Max Sequence Length: {self.config['max_len']}")
        
        # Model oluşturucu
        self.model_builder = SentimentModels(
            vocab_size=vocab_size,
            embedding_dim=self.config['embedding_dim'],
            max_len=self.config['max_len']
        )
        
        # Model seçimi
        model_builders = {
            'simple_rnn': self.model_builder.build_simple_rnn,
            'lstm': self.model_builder.build_lstm,
            'bilstm': self.model_builder.build_bilstm,
            'gru': self.model_builder.build_gru,
            'cnn': self.model_builder.build_cnn,
            'cnn_lstm': self.model_builder.build_cnn_lstm,
            'bilstm_attention': self.model_builder.build_bilstm_attention,
            'deep_lstm': self.model_builder.build_deep_lstm,
            'ensemble': self.model_builder.build_ensemble_model
        }
        
        if self.config['model_type'] not in model_builders:
            raise ValueError(f"Geçersiz model tipi: {self.config['model_type']}")
        
        # Model oluştur
        self.model = model_builders[self.config['model_type']]()
        
        # Model özeti
        print("\nModel Ozeti:")
        print("-"*50)
        self.model.summary()
        
        return self.model
    
    def train_model(self):
        """
        Modeli eğitir.
        """
        print("\n" + "="*60)
        print("ADIM 4: MODEL EGITIMI")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Önce model oluşturulmalı!")
        
        # Trainer
        self.trainer = ModelTrainer(
            model=self.model,
            save_dir=self.config['save_dir']
        )
        
        # Callbacks
        model_path = f"{self.config['save_dir']}/best_model.keras"
        callbacks = SentimentModels.get_callbacks(model_path, patience=5)
        
        # Eğitim
        self.history = self.trainer.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks
        )
        
        # Model tipi
        model_name = self.config['model_type']

        # Eğitim grafiği
        self.trainer.plot_training_history(
            save_path=f"{self.config['results_dir']}/{model_name}_training_history.png"
        )

        # Detaylı metrikler
        self.trainer.plot_detailed_metrics(
            save_path=f"{self.config['results_dir']}/{model_name}_detailed_metrics.png"
        )
        
        # Model kaydet
        self.trainer.save_model(f"{self.config['model_type']}_final")
        
        return self.history
    
    def evaluate_model(self):
        """
        Modeli değerlendirir.
        """
        print("\n" + "="*60)
        print("ADIM 5: MODEL DEGERLENDIRME")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Önce model eğitilmeli!")
        
        # Evaluator
        self.evaluator = ModelEvaluator(model=self.model)
        
        # Değerlendirme raporu
        self.evaluator.print_evaluation_report(
            self.X_test, self.y_test
        )
        
        # Model tipi
        model_name = self.config['model_type']

        # Confusion Matrix
        self.evaluator.plot_confusion_matrix(
            self.y_test,
            save_path=f"{self.config['results_dir']}/{model_name}_confusion_matrix.png"
        )

        # Normalized Confusion Matrix
        self.evaluator.plot_confusion_matrix(
            self.y_test,
            normalize=True,
            save_path=f"{self.config['results_dir']}/{model_name}_confusion_matrix_normalized.png"
        )

        # ROC Curve
        self.evaluator.plot_roc_curve(
            self.y_test,
            save_path=f"{self.config['results_dir']}/{model_name}_roc_curve.png"
        )

        # Precision-Recall Curve
        self.evaluator.plot_precision_recall_curve(
            self.y_test,
            save_path=f"{self.config['results_dir']}/{model_name}_pr_curve.png"
        )

        # Tahmin dağılımı
        self.evaluator.plot_prediction_distribution(
            self.y_test,
            save_path=f"{self.config['results_dir']}/{model_name}_prediction_distribution.png"
        )

        # Tüm metrikler
        self.evaluator.plot_all_metrics(
            self.y_test,
            save_dir=self.config['results_dir'],
            prefix=model_name
        )
        
        # Optimal threshold
        optimal_threshold, best_f1 = self.evaluator.find_optimal_threshold(
            self.y_test, metric='f1'
        )
        
        return self.evaluator.metrics
    
    def predict_sentiment(self, texts: list) -> list:
        """
        Yeni metinler için duygu tahmini yapar.
        
        Args:
            texts: Tahmin yapılacak metin listesi
            
        Returns:
            list: Tahmin sonuçları
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model ve preprocessor yüklü olmalı!")
        
        results = []
        for text in texts:
            # Ön işle
            processed = self.preprocessor.transform_single_text(
                text,
                remove_html=True,
                remove_special=True,
                lowercase=True
            )
            
            # Tahmin
            prob = self.model.predict(processed, verbose=0)[0][0]
            sentiment = 'positive' if prob >= 0.5 else 'negative'
            confidence = prob if prob >= 0.5 else 1 - prob
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': sentiment,
                'confidence': float(confidence),
                'probability': float(prob)
            })
        
        return results
    
    def run_full_pipeline(self, data_path: str = None):
        """
        Tam pipeline'ı çalıştırır.
        
        Args:
            data_path: Veri dosyası yolu
        """
        print("\n" + "="*60)
        print("   IMDB DUYGU ANALIZI - TAM PIPELINE")
        print("="*60 + "\n")
        
        # 1. Veri yükleme
        self.load_data(data_path)
        
        # 2. Ön işleme
        self.preprocess_data()
        
        # 3. Model oluşturma
        self.build_model()
        
        # 4. Eğitim
        self.train_model()
        
        # 5. Değerlendirme
        metrics = self.evaluate_model()
        
        # Sonuçları kaydet
        self.save_results()
        
        print("\n" + "="*60)
        print("PIPELINE TAMAMLANDI!")
        print("="*60)
        
        return metrics
    
    def save_results(self):
        """Tüm sonuçları kaydeder."""
        model_name = self.config['model_type']

        # Yapılandırma
        config_path = f"{self.config['results_dir']}/{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

        # Metrikler
        if self.evaluator and self.evaluator.metrics:
            metrics_path = f"{self.config['results_dir']}/{model_name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({k: float(v) for k, v in self.evaluator.metrics.items()}, f, indent=4)

        # Eğitim özeti
        if self.trainer:
            summary = self.trainer.get_training_summary()
            summary_path = f"{self.config['results_dir']}/{model_name}_training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump({k: float(v) if isinstance(v, (int, float)) else v
                          for k, v in summary.items()}, f, indent=4)

        print(f"\nSonuclar kaydedildi: {self.config['results_dir']}/{model_name}_*.json")
    
    def demo(self):
        """Demo tahminler yapar."""
        print("\n" + "="*60)
        print("DEMO TAHMINLER")
        print("="*60)
        
        sample_reviews = [
            "This movie was absolutely fantastic! Great acting and storyline.",
            "Terrible film. Waste of time and money. Do not watch!",
            "It was okay, nothing special but not bad either.",
            "One of the best movies I've ever seen! Highly recommended!",
            "Boring and predictable. I fell asleep halfway through."
        ]
        
        print("\nOrnek Yorumlar ve Tahminler:\n")

        results = self.predict_sentiment(sample_reviews)

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['text']}")
            print(f"   Tahmin: {result['sentiment'].upper()}")
            print(f"   Guven: {result['confidence']*100:.1f}%")
            print()


def train_multiple_models(pipeline: SentimentAnalysisPipeline, 
                         model_types: list = None) -> dict:
    """
    Birden fazla modeli eğitir ve karşılaştırır.
    
    Args:
        pipeline: Pipeline nesnesi
        model_types: Eğitilecek model tipleri
        
    Returns:
        dict: Model sonuçları
    """
    if model_types is None:
        model_types = ['lstm', 'bilstm', 'gru', 'cnn', 'cnn_lstm']
    
    results = {}
    
    print("\n" + "="*60)
    print("COKLU MODEL EGITIMI VE KARSILASTIRMA")
    print("="*60)
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Model: {model_type.upper()}")
        print(f"{'='*60}")
        
        # Model oluştur
        pipeline.build_model(model_type)
        
        # Eğit
        pipeline.train_model()
        
        # Değerlendir
        metrics = pipeline.evaluate_model()
        
        results[model_type] = metrics
    
    # Karşılaştırma grafiği
    compare_models(
        results,
        save_path=f"{pipeline.config['results_dir']}/model_comparison.png"
    )
    
    return results


def main():
    """Ana fonksiyon."""
    # Argüman ayrıştırıcı
    parser = argparse.ArgumentParser(
        description='IMDB Duygu Analizi - Derin Öğrenme Projesi'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/IMDB_Dataset.csv',
        help='Veri dosyası yolu'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='bilstm',
        choices=['simple_rnn', 'lstm', 'bilstm', 'gru', 'cnn', 
                'cnn_lstm', 'bilstm_attention', 'deep_lstm', 'ensemble'],
        help='Model tipi'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Epoch sayısı'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Batch boyutu'
    )
    
    parser.add_argument(
        '--max-words',
        type=int,
        default=10000,
        help='Maksimum kelime sayısı'
    )
    
    parser.add_argument(
        '--max-len',
        type=int,
        default=200,
        help='Maksimum dizi uzunluğu'
    )
    
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Birden fazla modeli karşılaştır'
    )
    
    args = parser.parse_args()
    
    # Yapılandırma
    config = {
        'data_path': args.data,
        'model_type': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'max_words': args.max_words,
        'max_len': args.max_len
    }
    
    # Pipeline oluştur
    pipeline = SentimentAnalysisPipeline(config)
    
    if args.compare:
        # Veri yükle ve ön işle
        pipeline.load_data()
        pipeline.preprocess_data()
        
        # Çoklu model karşılaştırması
        results = train_multiple_models(pipeline)
    else:
        # Tam pipeline
        pipeline.run_full_pipeline()
        
        # Demo
        pipeline.demo()


if __name__ == "__main__":
    main()

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import History

import warnings
warnings.filterwarnings('ignore')

# GPU yapılandırması
def configure_gpu():
    """GPU bellek büyümesini yapılandırır."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f" {len(gpus)} GPU bulundu ve yapılandırıldı.")
        except RuntimeError as e:
            print(f" GPU yapılandırma hatası: {e}")
    else:
        print(" GPU bulunamadı, CPU kullanılacak.")


class ModelTrainer:
    """
    Model eğitim yöneticisi sınıfı.
    
    Bu sınıf, model eğitimi, izleme ve kaydetme işlemlerini yönetir.
    
    Attributes:
        model (Model): Keras modeli
        history (History): Eğitim geçmişi
        config (dict): Eğitim yapılandırması
    """
    
    def __init__(self, model: Model = None, save_dir: str = 'models'):
        """
        ModelTrainer sınıfını başlatır.
        
        Args:
            model: Keras modeli
            save_dir: Model kayıt dizini
        """
        self.model = model
        self.history = None
        self.save_dir = save_dir
        self.training_time = None
        self.config = {}
        
        # Kayıt dizinini oluştur
        os.makedirs(save_dir, exist_ok=True)
        
        # GPU yapılandır
        configure_gpu()
    
    def set_model(self, model: Model):
        """
        Modeli ayarlar.
        
        Args:
            model: Keras modeli
        """
        self.model = model
        print(f" Model ayarlandı: {model.name}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 20, batch_size: int = 64,
              callbacks: list = None, verbose: int = 1) -> History:
        """
        Modeli eğitir.
        
        Args:
            X_train: Eğitim verileri
            y_train: Eğitim etiketleri
            X_val: Doğrulama verileri
            y_val: Doğrulama etiketleri
            epochs: Epoch sayısı
            batch_size: Batch boyutu
            callbacks: Callback listesi
            verbose: Çıktı detay seviyesi
            
        Returns:
            History: Eğitim geçmişi
        """
        if self.model is None:
            raise ValueError("Önce model ayarlanmalı!")
        
        print("\n" + "="*60)
        print(" MODEL EĞİTİMİ BAŞLIYOR")
        print("="*60)
        print(f"   Model: {self.model.name}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Eğitim Örnekleri: {len(X_train):,}")
        if X_val is not None:
            print(f"   Doğrulama Örnekleri: {len(X_val):,}")
        print("="*60 + "\n")
        
        # Yapılandırmayı kaydet
        self.config = {
            'model_name': self.model.name,
            'epochs': epochs,
            'batch_size': batch_size,
            'train_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else 0,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Validation verisi
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Eğitimi başlat
        start_time = datetime.now()
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Eğitim süresini hesapla
        self.training_time = (datetime.now() - start_time).total_seconds()
        self.config['training_time_seconds'] = self.training_time
        self.config['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print("\n" + "="*60)
        print(" EĞİTİM TAMAMLANDI!")
        print("="*60)
        print(f"   Toplam süre: {self.training_time:.2f} saniye ({self.training_time/60:.2f} dakika)")
        print(f"   Son train loss: {self.history.history['loss'][-1]:.4f}")
        print(f"   Son train accuracy: {self.history.history['accuracy'][-1]:.4f}")
        if X_val is not None:
            print(f"   Son val loss: {self.history.history['val_loss'][-1]:.4f}")
            print(f"   Son val accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        
        # En iyi değerleri bul
        if X_val is not None:
            best_epoch = np.argmin(self.history.history['val_loss']) + 1
            best_val_acc = max(self.history.history['val_accuracy'])
            print(f"\n    En iyi epoch: {best_epoch}")
            print(f"    En iyi val accuracy: {best_val_acc:.4f}")
            
            self.config['best_epoch'] = int(best_epoch)
            self.config['best_val_accuracy'] = float(best_val_acc)
        
        return self.history
    
    def plot_training_history(self, save_path: str = None, figsize: tuple = (14, 5)):
        """
        Eğitim geçmişini görselleştirir.
        
        Args:
            save_path: Grafik kayıt yolu
            figsize: Grafik boyutu
        """
        if self.history is None:
            raise ValueError("Önce model eğitilmeli!")
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss grafiği
        ax1 = axes[0]
        ax1.plot(epochs, history['loss'], 'b-', label='Eğitim Loss', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Doğrulama Loss', linewidth=2)
            best_epoch = np.argmin(history['val_loss']) + 1
            ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                       label=f'En iyi epoch ({best_epoch})')
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy grafiği
        ax2 = axes[1]
        ax2.plot(epochs, history['accuracy'], 'b-', label='Eğitim Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            ax2.plot(epochs, history['val_accuracy'], 'r-', label='Doğrulama Accuracy', linewidth=2)
            best_val_acc = max(history['val_accuracy'])
            ax2.axhline(y=best_val_acc, color='green', linestyle='--', alpha=0.7,
                       label=f'En iyi acc ({best_val_acc:.4f})')
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_detailed_metrics(self, save_path: str = None):
        """
        Detaylı eğitim metriklerini görselleştirir.
        
        Args:
            save_path: Grafik kayıt yolu
        """
        if self.history is None:
            raise ValueError("Önce model eğitilmeli!")
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Loss karşılaştırma
        ax1 = axes[0, 0]
        ax1.plot(epochs, history['loss'], 'b-o', label='Train', markersize=4)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-o', label='Validation', markersize=4)
        ax1.fill_between(epochs, history['loss'], alpha=0.2, color='blue')
        ax1.set_title('Loss Eğrisi', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy karşılaştırma
        ax2 = axes[0, 1]
        ax2.plot(epochs, history['accuracy'], 'b-o', label='Train', markersize=4)
        if 'val_accuracy' in history:
            ax2.plot(epochs, history['val_accuracy'], 'r-o', label='Validation', markersize=4)
        ax2.fill_between(epochs, history['accuracy'], alpha=0.2, color='blue')
        ax2.set_title('Accuracy Eğrisi', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Overfitting analizi (Train vs Val farkı)
        ax3 = axes[1, 0]
        if 'val_loss' in history:
            gap = np.array(history['val_loss']) - np.array(history['loss'])
            colors = ['red' if g > 0 else 'green' for g in gap]
            ax3.bar(epochs, gap, color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_title('Loss Farkı (Val - Train)\nPozitif = Overfitting', 
                         fontsize=12, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss Farkı')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Doğrulama verisi yok', ha='center', va='center')
        
        # 4. Learning rate (varsa)
        ax4 = axes[1, 1]
        if 'lr' in history:
            ax4.plot(epochs, history['lr'], 'g-o', markersize=4)
            ax4.set_title('Learning Rate', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('LR')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        else:
            # Epoch başına iyileşme
            if 'val_accuracy' in history:
                improvement = np.diff([0] + list(history['val_accuracy']))
                colors = ['green' if i > 0 else 'red' for i in improvement]
                ax4.bar(epochs, improvement, color=colors, alpha=0.7)
                ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax4.set_title('Epoch Başına Accuracy İyileşmesi', 
                             fontsize=12, fontweight='bold')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('İyileşme')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def save_model(self, filename: str = None, save_config: bool = True):
        """
        Modeli kaydeder.
        
        Args:
            filename: Dosya adı (uzantısız)
            save_config: Yapılandırmayı da kaydet
        """
        if self.model is None:
            raise ValueError("Kaydedilecek model yok!")
        
        if filename is None:
            filename = f"{self.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Model kaydet
        model_path = os.path.join(self.save_dir, f"{filename}.keras")
        self.model.save(model_path)
        print(f" Model kaydedildi: {model_path}")
        
        # Yapılandırmayı kaydet
        if save_config:
            config_path = os.path.join(self.save_dir, f"{filename}_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f" Yapılandırma kaydedildi: {config_path}")
        
        # Eğitim geçmişini kaydet
        if self.history is not None:
            history_path = os.path.join(self.save_dir, f"{filename}_history.pkl")
            with open(history_path, 'wb') as f:
                pickle.dump(self.history.history, f)
            print(f" Eğitim geçmişi kaydedildi: {history_path}")
        
        return model_path
    
    def load_model(self, model_path: str):
        """
        Modeli yükler.
        
        Args:
            model_path: Model dosya yolu
        """
        self.model = load_model(model_path)
        print(f" Model yüklendi: {model_path}")
        
        # Yapılandırmayı yükle
        config_path = model_path.replace('.keras', '_config.json').replace('.h5', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f" Yapılandırma yüklendi")
        
        # Geçmişi yükle
        history_path = model_path.replace('.keras', '_history.pkl').replace('.h5', '_history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history_dict = pickle.load(f)
            # History nesnesini oluştur
            class HistoryLike:
                def __init__(self, history_dict):
                    self.history = history_dict
            self.history = HistoryLike(history_dict)
            print(f" Eğitim geçmişi yüklendi")
        
        return self.model
    
    def get_training_summary(self) -> dict:
        """
        Eğitim özetini döndürür.
        
        Returns:
            dict: Eğitim özeti
        """
        if self.history is None:
            return {}
        
        history = self.history.history
        
        summary = {
            'model_name': self.model.name if self.model else 'Unknown',
            'total_epochs': len(history['loss']),
            'final_train_loss': history['loss'][-1],
            'final_train_accuracy': history['accuracy'][-1],
            'min_train_loss': min(history['loss']),
            'max_train_accuracy': max(history['accuracy']),
        }
        
        if 'val_loss' in history:
            summary.update({
                'final_val_loss': history['val_loss'][-1],
                'final_val_accuracy': history['val_accuracy'][-1],
                'min_val_loss': min(history['val_loss']),
                'max_val_accuracy': max(history['val_accuracy']),
                'best_epoch': int(np.argmin(history['val_loss']) + 1)
            })
        
        if self.training_time:
            summary['training_time_seconds'] = self.training_time
            summary['training_time_minutes'] = self.training_time / 60
        
        return summary
    
    def print_training_summary(self):
        """Eğitim özetini yazdırır."""
        summary = self.get_training_summary()
        
        if not summary:
            print(" Eğitim özeti bulunamadı!")
            return
        
        print("\n" + "="*60)
        print(" EĞİTİM ÖZETİ")
        print("="*60)
        
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print("="*60)


# Test kodu
if __name__ == "__main__":
    print(" Trainer modülü test ediliyor...")
    
    # GPU yapılandırma testi
    configure_gpu()
    
    # Trainer oluştur
    trainer = ModelTrainer(save_dir='test_models')
    
    print("\n Trainer modülü testi başarılı!")

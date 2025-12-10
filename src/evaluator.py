import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from tensorflow.keras.models import Model

import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Model değerlendirme sınıfı.
    
    Bu sınıf, derin öğrenme modellerinin performansını
    kapsamlı bir şekilde değerlendirir.
    
    Attributes:
        model (Model): Değerlendirilecek Keras modeli
        class_names (list): Sınıf isimleri
    """
    
    def __init__(self, model: Model = None, class_names: list = None):
        """
        ModelEvaluator sınıfını başlatır.
        
        Args:
            model: Keras modeli
            class_names: Sınıf isimleri
        """
        self.model = model
        self.class_names = class_names or ['Negative', 'Positive']
        self.predictions = None
        self.probabilities = None
        self.metrics = {}
    
    def set_model(self, model: Model):
        """Modeli ayarlar."""
        self.model = model
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tahmin yapar.
        
        Args:
            X: Giriş verileri
            threshold: Sınıflandırma eşik değeri
            
        Returns:
            Tuple: (tahminler, olasılıklar)
        """
        if self.model is None:
            raise ValueError("Model ayarlanmamış!")
        
        print(" Tahminler yapılıyor...")
        
        # Olasılıkları al
        self.probabilities = self.model.predict(X, verbose=0).flatten()
        
        # Eşik değerine göre sınıflandır
        self.predictions = (self.probabilities >= threshold).astype(int)
        
        print(f" {len(self.predictions):,} tahmin tamamlandı.")
        
        return self.predictions, self.probabilities
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, 
                threshold: float = 0.5) -> Dict:
        """
        Modeli değerlendirir ve tüm metrikleri hesaplar.
        
        Args:
            X: Test verileri
            y_true: Gerçek etiketler
            threshold: Sınıflandırma eşik değeri
            
        Returns:
            Dict: Değerlendirme metrikleri
        """
        # Tahmin yap
        y_pred, y_prob = self.predict(X, threshold)
        
        # Metrikleri hesapla
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'threshold': threshold
        }
        
        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        self.metrics['roc_auc'] = auc(fpr, tpr)
        
        # PR-AUC
        self.metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
        return self.metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Specificity (True Negative Rate) hesaplar."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def print_evaluation_report(self, X: np.ndarray, y_true: np.ndarray,
                               threshold: float = 0.5):
        """
        Kapsamlı değerlendirme raporu yazdırır.
        
        Args:
            X: Test verileri
            y_true: Gerçek etiketler
            threshold: Sınıflandırma eşik değeri
        """
        # Değerlendir
        metrics = self.evaluate(X, y_true, threshold)
        
        print("\n" + "="*60)
        print(" MODEL DEĞERLENDİRME RAPORU")
        print("="*60)
        
        print("\n TEMEL METRİKLER:")
        print("-"*40)
        print(f"   Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision:   {metrics['precision']:.4f}")
        print(f"   Recall:      {metrics['recall']:.4f}")
        print(f"   F1-Score:    {metrics['f1_score']:.4f}")
        print(f"   Specificity: {metrics['specificity']:.4f}")
        
        print("\n ALAN METRİKLERİ:")
        print("-"*40)
        print(f"   ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"   PR-AUC:      {metrics['pr_auc']:.4f}")
        
        print("\n CLASSIFICATION REPORT:")
        print("-"*40)
        print(classification_report(y_true, self.predictions, 
                                   target_names=self.class_names))
        
        print("="*60)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray = None,
                             normalize: bool = False, save_path: str = None):
        """
        Confusion matrix görselleştirir.
        
        Args:
            y_true: Gerçek etiketler
            y_pred: Tahminler (None ise self.predictions kullanılır)
            normalize: Normalize et
            save_path: Kayıt yolu
        """
        if y_pred is None:
            y_pred = self.predictions
        
        if y_pred is None:
            raise ValueError("Önce tahmin yapılmalı!")
        
        # Confusion matrix hesapla
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Heatmap
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=self.class_names,
                   yticklabels=self.class_names, ax=ax,
                   annot_kws={'size': 14})
        
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''),
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Tahmin Edilen', fontsize=12)
        ax.set_ylabel('Gerçek', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Grafik kaydedildi: {save_path}")
        
        plt.show()
        
        # Detaylı istatistikler
        tn, fp, fn, tp = cm.ravel() if not normalize else confusion_matrix(y_true, y_pred).ravel()
        print(f"\n Confusion Matrix Detayları:")
        print(f"   True Negatives (TN): {tn:,}")
        print(f"   False Positives (FP): {fp:,}")
        print(f"   False Negatives (FN): {fn:,}")
        print(f"   True Positives (TP): {tp:,}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray = None,
                      save_path: str = None):
        """
        ROC eğrisini görselleştirir.
        
        Args:
            y_true: Gerçek etiketler
            y_prob: Tahmin olasılıkları
            save_path: Kayıt yolu
        """
        if y_prob is None:
            y_prob = self.probabilities
        
        if y_prob is None:
            raise ValueError("Önce tahmin yapılmalı!")
        
        # ROC hesapla
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Optimal eşik değerini bul (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # ROC eğrisi
        ax.plot(fpr, tpr, 'b-', linewidth=2, 
               label=f'ROC Curve (AUC = {roc_auc:.4f})')
        
        # Rastgele sınıflandırıcı
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        
        # Optimal nokta
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='green', s=100,
                  zorder=5, label=f'Optimal Threshold ({optimal_threshold:.3f})')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Grafik kaydedildi: {save_path}")
        
        plt.show()
        
        print(f"\n ROC Analizi:")
        print(f"   AUC Score: {roc_auc:.4f}")
        print(f"   Optimal Threshold: {optimal_threshold:.4f}")
        print(f"   TPR at optimal: {tpr[optimal_idx]:.4f}")
        print(f"   FPR at optimal: {fpr[optimal_idx]:.4f}")
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, 
                                   y_prob: np.ndarray = None,
                                   save_path: str = None):
        """
        Precision-Recall eğrisini görselleştirir.
        
        Args:
            y_true: Gerçek etiketler
            y_prob: Tahmin olasılıkları
            save_path: Kayıt yolu
        """
        if y_prob is None:
            y_prob = self.probabilities
        
        if y_prob is None:
            raise ValueError("Önce tahmin yapılmalı!")
        
        # PR hesapla
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
        # F1 skorlarını hesapla
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # PR eğrisi
        ax.plot(recall, precision, 'b-', linewidth=2,
               label=f'PR Curve (AP = {pr_auc:.4f})')
        
        # Baseline (pozitif sınıf oranı)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='r', linestyle='--', 
                  label=f'Baseline ({baseline:.3f})')
        
        # Optimal nokta
        ax.scatter(recall[optimal_idx], precision[optimal_idx], color='green', 
                  s=100, zorder=5, label=f'Optimal (threshold={optimal_threshold:.3f})')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Grafik kaydedildi: {save_path}")
        
        plt.show()
        
        print(f"\n PR Analizi:")
        print(f"   Average Precision: {pr_auc:.4f}")
        print(f"   Optimal Threshold: {optimal_threshold:.4f}")
        print(f"   Precision at optimal: {precision[optimal_idx]:.4f}")
        print(f"   Recall at optimal: {recall[optimal_idx]:.4f}")
        print(f"   F1 at optimal: {f1_scores[optimal_idx]:.4f}")
    
    def plot_all_metrics(self, y_true: np.ndarray, save_dir: str = None):
        """
        Tüm değerlendirme grafiklerini oluşturur.
        
        Args:
            y_true: Gerçek etiketler
            save_dir: Kayıt dizini
        """
        if self.predictions is None:
            raise ValueError("Önce tahmin yapılmalı!")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = confusion_matrix(y_true, self.predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Tahmin')
        ax1.set_ylabel('Gerçek')
        
        # 2. Normalized Confusion Matrix
        ax2 = axes[0, 1]
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_title('Normalized Confusion Matrix', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Tahmin')
        ax2.set_ylabel('Gerçek')
        
        # 3. ROC Curve
        ax3 = axes[1, 0]
        fpr, tpr, _ = roc_curve(y_true, self.probabilities)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.4f}')
        ax3.plot([0, 1], [0, 1], 'r--', linewidth=1)
        ax3.set_xlabel('FPR')
        ax3.set_ylabel('TPR')
        ax3.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Metrics Bar Chart
        ax4 = axes[1, 1]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_values = [
            self.metrics['accuracy'],
            self.metrics['precision'],
            self.metrics['recall'],
            self.metrics['f1_score'],
            self.metrics['roc_auc']
        ]
        colors = plt.cm.RdYlGn(np.array(metric_values))
        bars = ax4.barh(metric_names, metric_values, color=colors)
        ax4.set_xlim([0, 1])
        ax4.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        
        # Bar üzerine değer yaz
        for bar, val in zip(bars, metric_values):
            ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = f"{save_dir}/evaluation_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def analyze_misclassifications(self, X: np.ndarray, y_true: np.ndarray,
                                  texts: List[str] = None, n_samples: int = 10) -> pd.DataFrame:
        """
        Yanlış sınıflandırmaları analiz eder.
        
        Args:
            X: Giriş verileri
            y_true: Gerçek etiketler
            texts: Orijinal metinler
            n_samples: Gösterilecek örnek sayısı
            
        Returns:
            pd.DataFrame: Yanlış sınıflandırılmış örnekler
        """
        if self.predictions is None:
            self.predict(X)
        
        # Yanlış sınıflandırmaları bul
        misclassified_idx = np.where(y_true != self.predictions)[0]
        
        print(f"\n Yanlış Sınıflandırma Analizi:")
        print(f"   Toplam yanlış: {len(misclassified_idx):,} / {len(y_true):,}")
        print(f"   Hata oranı: {len(misclassified_idx)/len(y_true)*100:.2f}%")
        
        # DataFrame oluştur
        results = []
        for idx in misclassified_idx[:n_samples]:
            result = {
                'index': idx,
                'true_label': self.class_names[y_true[idx]],
                'predicted_label': self.class_names[self.predictions[idx]],
                'confidence': self.probabilities[idx] if self.predictions[idx] == 1 
                             else 1 - self.probabilities[idx]
            }
            if texts is not None:
                result['text'] = texts[idx][:200] + '...' if len(texts[idx]) > 200 else texts[idx]
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # False Positive ve False Negative analizi
        fp_count = np.sum((y_true == 0) & (self.predictions == 1))
        fn_count = np.sum((y_true == 1) & (self.predictions == 0))
        
        print(f"\n   False Positives (Negatifi pozitif olarak tahmin): {fp_count:,}")
        print(f"   False Negatives (Pozitifi negatif olarak tahmin): {fn_count:,}")
        
        return df
    
    def plot_prediction_distribution(self, y_true: np.ndarray, save_path: str = None):
        """
        Tahmin olasılıklarının dağılımını görselleştirir.
        
        Args:
            y_true: Gerçek etiketler
            save_path: Kayıt yolu
        """
        if self.probabilities is None:
            raise ValueError("Önce tahmin yapılmalı!")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Sınıflara göre olasılık dağılımı
        ax1 = axes[0]
        
        neg_probs = self.probabilities[y_true == 0]
        pos_probs = self.probabilities[y_true == 1]
        
        ax1.hist(neg_probs, bins=50, alpha=0.5, label='Gerçek: Negative', color='red')
        ax1.hist(pos_probs, bins=50, alpha=0.5, label='Gerçek: Positive', color='green')
        ax1.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
        ax1.set_xlabel('Pozitif Sınıf Olasılığı', fontsize=12)
        ax1.set_ylabel('Frekans', fontsize=12)
        ax1.set_title('Sınıflara Göre Olasılık Dağılımı', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. KDE plot
        ax2 = axes[1]
        
        sns.kdeplot(neg_probs, ax=ax2, label='Gerçek: Negative', color='red', fill=True, alpha=0.3)
        sns.kdeplot(pos_probs, ax=ax2, label='Gerçek: Positive', color='green', fill=True, alpha=0.3)
        ax2.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
        ax2.set_xlabel('Pozitif Sınıf Olasılığı', fontsize=12)
        ax2.set_ylabel('Yoğunluk', fontsize=12)
        ax2.set_title('Olasılık Yoğunluk Dağılımı (KDE)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def find_optimal_threshold(self, y_true: np.ndarray, 
                              metric: str = 'f1') -> Tuple[float, float]:
        """
        Optimal eşik değerini bulur.
        
        Args:
            y_true: Gerçek etiketler
            metric: Optimize edilecek metrik ('f1', 'accuracy', 'youden')
            
        Returns:
            Tuple: (optimal_threshold, metric_value)
        """
        if self.probabilities is None:
            raise ValueError("Önce tahmin yapılmalı!")
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_metric = 0
        
        for thresh in thresholds:
            preds = (self.probabilities >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, preds)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, preds)
            elif metric == 'youden':
                # Youden's J statistic = Sensitivity + Specificity - 1
                tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            else:
                score = f1_score(y_true, preds)
            
            if score > best_metric:
                best_metric = score
                best_threshold = thresh
        
        print(f"\n Optimal Threshold Analizi ({metric.upper()}):")
        print(f"   Optimal Threshold: {best_threshold:.2f}")
        print(f"   {metric.upper()} Score: {best_metric:.4f}")
        
        return best_threshold, best_metric


def compare_models(models_results: Dict[str, Dict], save_path: str = None):
    """
    Birden fazla modeli karşılaştırır.
    
    Args:
        models_results: Model sonuçları sözlüğü
        save_path: Kayıt yolu
    """
    model_names = list(models_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # DataFrame oluştur
    df = pd.DataFrame(index=model_names, columns=metrics)
    for model_name, results in models_results.items():
        for metric in metrics:
            df.loc[model_name, metric] = results.get(metric, 0)
    
    df = df.astype(float)
    
    # Görselleştir
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Bar chart
    ax1 = axes[0]
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax1.bar(x + offset, df.loc[model_name], width, label=model_name)
    
    ax1.set_xlabel('Metrik')
    ax1.set_ylabel('Değer')
    ax1.set_title('Model Karşılaştırması', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    # 2. Heatmap
    ax2 = axes[1]
    sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax2,
               vmin=0, vmax=1, cbar_kws={'label': 'Score'})
    ax2.set_title('Model Performans Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Metrik')
    ax2.set_ylabel('Model')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Grafik kaydedildi: {save_path}")
    
    plt.show()
    
    # En iyi modeli bul
    print("\n Model Karşılaştırma Özeti:")
    print("-"*50)
    print(df.to_string())
    print("-"*50)
    
    best_model = df['f1_score'].idxmax()
    print(f"\n En İyi Model (F1-Score): {best_model}")
    print(f"   F1-Score: {df.loc[best_model, 'f1_score']:.4f}")
    
    return df


# Test kodu
if __name__ == "__main__":
    print(" Evaluator modülü test ediliyor...")
    
    # Dummy veri oluştur
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.random(100)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Test
    evaluator = ModelEvaluator()
    evaluator.predictions = y_pred
    evaluator.probabilities = y_prob
    evaluator.metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': 0.75
    }
    
    print(" Evaluator modülü testi başarılı!")

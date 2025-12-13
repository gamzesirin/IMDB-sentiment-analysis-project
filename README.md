# IMDB Duygu Analizi - Derin Öğrenme Projesi

## İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Model Mimarileri](#-model-mimarileri)
- [Proje Yapısı](#-proje-yapısı)
- [Sonuçlar](#-sonuçlar)

---

## Proje Hakkında

Bu proje, **Derin Öğrenme** yüksek lisans dersi final projesi kapsamında hazırlanmıştır. IMDB film yorumlarından duygu analizi yapan çeşitli derin öğrenme modelleri geliştirilmiş ve karşılaştırılmıştır.

### Problem Tanımı

Duygu analizi (Sentiment Analysis), doğal dil işlemenin (NLP) önemli bir alt alanıdır. Bu projede, film yorumlarının pozitif veya negatif olarak sınıflandırılması amaçlanmaktadır.

### Veri Seti

- **Kaynak:** [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Boyut:** 50,000 film yorumu
- **Sınıflar:** Pozitif (25,000) ve Negatif (25,000)

---

## Özellikler

- **Kapsamlı Veri Analizi:** Keşifsel veri analizi ve görselleştirmeler
- **Metin Ön İşleme:** HTML temizleme, tokenization, padding
- **Çoklu Model Mimarileri:** RNN, LSTM, BiLSTM, GRU, CNN, Hybrid
- **Detaylı Değerlendirme:** Confusion Matrix, ROC Curve, PR Curve
- **Model Karşılaştırması:** Farklı modellerin performans karşılaştırması
- **Model Kaydetme/Yükleme:** Eğitilmiş modelleri kaydetme ve yükleme
- **Demo Tahminler:** Yeni metinler için duygu tahmini

---

## Kurulum

### Gereksinimler

- Python 3.8+
- TensorFlow 2.10+

### Adım 1: Projeyi İndirin

```bash
git clone https://github.com/gamzesirin/IMDB-sentiment-analysis.git
cd IMDB-sentiment-analysis
```

### Adım 3: Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### Adım 4: Veriyi İndirin

Kaggle'dan veri setini indirin ve `data/` klasörüne yerleştirin:

```bash
# Kaggle CLI kullanarak
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
unzip imdb-dataset-of-50k-movie-reviews.zip -d data/
```

---

## Kullanım

### Komut Satırından Çalıştırma

```bash
# Varsayılan ayarlarla (BiLSTM modeli)
python main.py --data data/IMDB\ Dataset.csv

# Farklı bir model ile
python main.py --model lstm --epochs 15 --batch-size 32

# Model karşılaştırması
python main.py --compare
```

### Komut Satırı Parametreleri

| Parametre      | Kısa | Varsayılan              | Açıklama                    |
| -------------- | ---- | ----------------------- | --------------------------- |
| `--data`       | `-d` | `data/IMDB Dataset.csv` | Veri dosyası yolu           |
| `--model`      | `-m` | `bilstm`                | Model tipi                  |
| `--epochs`     | `-e` | `10`                    | Epoch sayısı                |
| `--batch-size` | `-b` | `64`                    | Batch boyutu                |
| `--max-words`  | -    | `10000`                 | Kelime haznesi boyutu       |
| `--max-len`    | -    | `200`                   | Maksimum dizi uzunluğu      |
| `--compare`    | `-c` | `False`                 | Çoklu model karşılaştırması |

### Jupyter Notebook

```bash
jupyter notebook IMDB_Sentiment_Analysis.ipynb
```

### Python'da Kullanım

```python
from src.data_loader import IMDBDataLoader
from src.preprocessor import TextPreprocessor
from src.models import SentimentModels
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator

# Veri yükle
loader = IMDBDataLoader('data/IMDB Dataset.csv')
df = loader.load_data()

# Ön işle
preprocessor = TextPreprocessor(max_words=10000, max_len=200)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(df)

# Model oluştur
model_builder = SentimentModels(vocab_size=10001, embedding_dim=128, max_len=200)
model = model_builder.build_bilstm()

# Eğit
trainer = ModelTrainer(model=model)
trainer.train(X_train, y_train, X_val, y_val, epochs=10)

# Değerlendir
evaluator = ModelEvaluator(model=model)
evaluator.print_evaluation_report(X_test, y_test)
```

---

## Model Mimarileri

### 1. Simple RNN

```
Embedding → SpatialDropout → SimpleRNN → Dropout → Dense → Sigmoid
```

### 2. LSTM

```
Embedding → SpatialDropout → LSTM → Dense → Dropout → Sigmoid
```

### 3. Bidirectional LSTM (BiLSTM)

```
Embedding → SpatialDropout → BiLSTM → Dropout → BiLSTM → Dense → Sigmoid
```

### 4. GRU

```
Embedding → SpatialDropout → GRU → Dropout → GRU → Dense → Sigmoid
```

### 5. CNN (TextCNN)

```
Embedding → Conv1D(3,4,5) → GlobalMaxPool → Concatenate → Dense → Sigmoid
```

### 6. CNN + LSTM (Hybrid)

```
Embedding → Conv1D → MaxPool → Conv1D → MaxPool → LSTM → Dense → Sigmoid
```

### 7. BiLSTM + Attention

```
Embedding → BiLSTM → Self-Attention → GlobalAvgPool → Dense → Sigmoid
```

---

## Proje Yapısı

```
imdb_sentiment_analysis/
│
├── data/                          # Veri dosyaları
│   └── IMDB Dataset.csv          # Ham veri
│
├── models/                        # Eğitilmiş modeller
│   ├── best_model.keras          # En iyi model
│   ├── tokenizer.pkl             # Tokenizer
│
├── results/                       # Sonuç grafikleri
│   ├── sentiment_distribution.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── model_comparison.png
│
├── src/                           # Kaynak kodlar
│   ├── __init__.py
│   ├── data_loader.py            # Veri yükleme
│   ├── preprocessor.py           # Metin ön işleme
│   ├── models.py                 # Model mimarileri
│   ├── trainer.py                # Eğitim fonksiyonları
│   └── evaluator.py              # Değerlendirme
│
├── main.py                        # Ana çalıştırma dosyası
├── requirements.txt               # Bağımlılıklar
└── README.md                      # Bu dosya
```

---

## Sonuçlar

### Model Karşılaştırması

| Model    | Accuracy | Precision | Recall   | F1-Score | ROC-AUC  |
| -------- | -------- | --------- | -------- | -------- | -------- |
| LSTM     | 0.87     | 0.86      | 0.88     | 0.87     | 0.94     |
| BiLSTM   | **0.89** | **0.88**  | **0.90** | **0.89** | **0.95** |
| GRU      | 0.87     | 0.86      | 0.88     | 0.87     | 0.94     |
| CNN      | 0.86     | 0.85      | 0.87     | 0.86     | 0.93     |
| CNN+LSTM | 0.88     | 0.87      | 0.89     | 0.88     | 0.94     |

### Temel Bulgular

1. **BiLSTM modeli en iyi performansı gösterdi** - Çift yönlü LSTM, metni hem ileri hem geri yönde işleyerek bağlamı daha iyi anlar.

2. **CNN modelleri hızlı ama biraz düşük performanslı** - Yerel özellikleri yakalama konusunda iyi, ancak uzun vadeli bağımlılıkları öğrenmekte zorlanıyor.

3. **Attention mekanizması faydalı** - Model, önemli kelimelere odaklanabiliyor.

---

## Hiperparametreler

| Parametre     | Değer  | Açıklama               |
| ------------- | ------ | ---------------------- |
| MAX_WORDS     | 10,000 | Kelime haznesi boyutu  |
| MAX_LEN       | 200    | Maksimum dizi uzunluğu |
| EMBEDDING_DIM | 128    | Embedding boyutu       |
| BATCH_SIZE    | 64     | Batch boyutu           |
| EPOCHS        | 10-15  | Epoch sayısı           |
| DROPOUT       | 0.5    | Dropout oranı          |
| LEARNING_RATE | 0.001  | Öğrenme oranı          |

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**

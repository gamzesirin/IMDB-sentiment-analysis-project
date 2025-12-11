import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Proje dizinini ekle
sys.path.insert(0, '/home/claude/imdb_sentiment_analysis')

print("="*60)
print("IMDB DUYGU ANALİZİ - PROJE TESTİ")
print("="*60)

# 1. Modülleri test et
print("\n 1. Modüller yükleniyor...")
try:
    from src.data_loader import IMDBDataLoader
    from src.preprocessor import TextPreprocessor
    from src.models import SentimentModels
    from src.trainer import ModelTrainer
    from src.evaluator import ModelEvaluator
    print("Tüm modüller başarıyla yüklendi!")
except Exception as e:
    print(f"Modül yükleme hatası: {e}")
    sys.exit(1)

# 2. Veri yükleme testi
print("\n 2. Veri yükleniyor...")
try:
    loader = IMDBDataLoader('data/IMDB_Dataset.csv')
    df = loader.load_data()
    print(f"Veri yüklendi: {len(df)} örnek")
except Exception as e:
    print(f"Veri yükleme hatası: {e}")
    sys.exit(1)

# 3. Ön işleme testi
print("\n[3] On isleme yapiliyor...")
try:
    preprocessor = TextPreprocessor(max_words=1000, max_len=100)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(
        df,
        text_column='review',
        label_column='sentiment',
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    print(f"Ön işleme tamamlandı!")
    print(f"Eğitim: {len(X_train)}, Doğrulama: {len(X_val)}, Test: {len(X_test)}")
except Exception as e:
    print(f" Ön işleme hatası: {e}")
    sys.exit(1)

# 4. Model oluşturma testi
print("\n 4. Model oluşturuluyor...")
try:
    vocab_size = preprocessor.get_vocabulary_size()
    model_builder = SentimentModels(vocab_size=vocab_size, embedding_dim=32, max_len=100)
    model = model_builder.build_lstm(units=32, dropout=0.3)
    # Model'i build et
    model.build(input_shape=(None, 100))
    print(f"Model oluşturuldu: {model.name}")
    print(f"      Parametre sayısı: {model.count_params():,}")
except Exception as e:
    print(f"Model oluşturma hatası: {e}")
    sys.exit(1)

# 5. Eğitim testi
print("\n[5] Model egitiliyor (2 epoch)...")
try:
    trainer = ModelTrainer(model=model, save_dir='models')
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=2,
        batch_size=8,
        verbose=0
    )
    print(f"Egitim tamamlandi!")
    print(f"      Son loss: {history.history['loss'][-1]:.4f}")
    print(f"      Son accuracy: {history.history['accuracy'][-1]:.4f}")
except Exception as e:
    print(f"Egitim hatasi: {e}")
    sys.exit(1)

# 6. Değerlendirme testi
print("\n6. Model değerlendiriliyor...")
try:
    evaluator = ModelEvaluator(model=model)
    metrics = evaluator.evaluate(X_test, y_test)
    print(f"Degerlendirme tamamlandi!")
    print(f"      Accuracy: {metrics['accuracy']:.4f}")
    print(f"      F1-Score: {metrics['f1_score']:.4f}")
    print(f"      ROC-AUC: {metrics['roc_auc']:.4f}")
except Exception as e:
    print(f"Degerlendirme hatasi: {e}")
    sys.exit(1)

# 7. Tahmin testi
print("\n[7] Tahmin yapiliyor...")
try:
    test_texts = [
        "This movie was amazing! I loved every moment.",
        "Terrible film, complete waste of time."
    ]
    
    for text in test_texts:
        processed = preprocessor.transform_single_text(text)
        prob = model.predict(processed, verbose=0)[0][0]
        sentiment = "Pozitif" if prob >= 0.5 else "Negatif"
        print(f"   \"{text[:40]}...\"")
        print(f"      -> {sentiment} (guven: {max(prob, 1-prob)*100:.1f}%)")
    print("Tahminler tamamlandi!")
except Exception as e:
    print(f"Tahmin hatasi: {e}")
    sys.exit(1)

# Final
print("\n" + "="*60)
print("TUM TESTLER BASARIYLA TAMAMLANDI!")
print("="*60)
print("\nProje kullanima hazir!")
print("   - Jupyter Notebook: IMDB_Sentiment_Analysis.ipynb")
print("   - Komut satiri: python main.py --help")
print("="*60)

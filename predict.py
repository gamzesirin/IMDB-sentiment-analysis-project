from src.preprocessor import TextPreprocessor
from tensorflow.keras.models import load_model

# Model ve tokenizer yükle
model = load_model('models/bilstm_final.keras')
preprocessor = TextPreprocessor()
preprocessor.load_tokenizer('models/tokenizer.pkl')

# Test metni
text = 'This movie was absolutely amazing! Great acting and story.'

# Tahmin yap
padded = preprocessor.transform_single_text(text)
pred = model.predict(padded, verbose=0)[0][0]

# Sonucu yazdır
sentiment = "Pozitif" if pred > 0.5 else "Negatif"
print(f'Tahmin: {sentiment} ({pred:.2%})')

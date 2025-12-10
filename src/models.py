import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, Dense, Dropout, LSTM, GRU, SimpleRNN,
    Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D,
    MaxPooling1D, Flatten, Input, Concatenate, BatchNormalization,
    SpatialDropout1D, Layer, Attention, MultiHeadAttention,
    LayerNormalization, Add
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

import warnings
warnings.filterwarnings('ignore')


class SentimentModels:
    """
    Duygu analizi için çeşitli derin öğrenme modelleri.
    
    Bu sınıf, farklı mimarileri kolayca oluşturmak ve karşılaştırmak
    için tasarlanmıştır.
    
    Attributes:
        vocab_size (int): Kelime haznesi boyutu
        embedding_dim (int): Embedding vektör boyutu
        max_len (int): Maksimum dizi uzunluğu
        embedding_matrix (np.ndarray): Önceden eğitilmiş embedding matrisi
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, max_len: int = 200,
                 embedding_matrix: np.ndarray = None):
        """
        SentimentModels sınıfını başlatır.
        
        Args:
            vocab_size: Kelime haznesi boyutu
            embedding_dim: Embedding vektör boyutu
            max_len: Maksimum dizi uzunluğu
            embedding_matrix: Önceden eğitilmiş embedding matrisi (opsiyonel)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        
    def _get_embedding_layer(self, trainable: bool = True) -> Embedding:
        """
        Embedding katmanı oluşturur.
        
        Args:
            trainable: Embedding ağırlıkları eğitilebilir mi?
            
        Returns:
            Embedding: Keras Embedding katmanı
        """
        if self.embedding_matrix is not None:
            return Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                weights=[self.embedding_matrix],
                input_length=self.max_len,
                trainable=trainable,
                name='embedding'
            )
        else:
            return Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                name='embedding'
            )
    
    def build_simple_rnn(self, units: int = 64, dropout: float = 0.5) -> Sequential:
        """
        Basit RNN modeli oluşturur.
        
        Args:
            units: RNN birim sayısı
            dropout: Dropout oranı
            
        Returns:
            Sequential: Derlenmiş model
        """
        model = Sequential([
            self._get_embedding_layer(),
            SpatialDropout1D(0.2),
            SimpleRNN(units, return_sequences=False),
            Dropout(dropout),
            Dense(64, activation='relu'),
            Dropout(dropout/2),
            Dense(1, activation='sigmoid')
        ], name='SimpleRNN_Model')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_lstm(self, units: int = 128, dropout: float = 0.5,
                   recurrent_dropout: float = 0.2) -> Sequential:
        """
        LSTM modeli oluşturur.
        
        LSTM (Long Short-Term Memory), uzun vadeli bağımlılıkları
        öğrenebilen bir RNN varyantıdır.
        
        Args:
            units: LSTM birim sayısı
            dropout: Dropout oranı
            recurrent_dropout: Recurrent dropout oranı
            
        Returns:
            Sequential: Derlenmiş model
        """
        model = Sequential([
            self._get_embedding_layer(),
            SpatialDropout1D(0.2),
            LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout),
            Dense(64, activation='relu'),
            Dropout(dropout/2),
            Dense(1, activation='sigmoid')
        ], name='LSTM_Model')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_bilstm(self, units: int = 64, dropout: float = 0.5) -> Sequential:
        """
        Bidirectional LSTM modeli oluşturur.
        
        BiLSTM, metni hem ileri hem de geri yönde işleyerek
        bağlamı daha iyi anlar.
        
        Args:
            units: LSTM birim sayısı
            dropout: Dropout oranı
            
        Returns:
            Sequential: Derlenmiş model
        """
        model = Sequential([
            self._get_embedding_layer(),
            SpatialDropout1D(0.2),
            Bidirectional(LSTM(units, return_sequences=True)),
            Dropout(dropout),
            Bidirectional(LSTM(units // 2)),
            Dense(64, activation='relu'),
            Dropout(dropout/2),
            Dense(1, activation='sigmoid')
        ], name='BiLSTM_Model')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_gru(self, units: int = 128, dropout: float = 0.5) -> Sequential:
        """
        GRU modeli oluşturur.
        
        GRU (Gated Recurrent Unit), LSTM'e benzer ancak daha az
        parametre ile çalışır.
        
        Args:
            units: GRU birim sayısı
            dropout: Dropout oranı
            
        Returns:
            Sequential: Derlenmiş model
        """
        model = Sequential([
            self._get_embedding_layer(),
            SpatialDropout1D(0.2),
            GRU(units, return_sequences=True),
            Dropout(dropout),
            GRU(units // 2),
            Dense(64, activation='relu'),
            Dropout(dropout/2),
            Dense(1, activation='sigmoid')
        ], name='GRU_Model')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn(self, filters: int = 128, kernel_sizes: list = [3, 4, 5],
                  dropout: float = 0.5) -> Model:
        """
        CNN modeli oluşturur (TextCNN).
        
        Farklı boyutlarda filtreler kullanarak n-gram benzeri
        özellikleri yakalar.
        
        Args:
            filters: Her filtre boyutu için filtre sayısı
            kernel_sizes: Filtre boyutları listesi
            dropout: Dropout oranı
            
        Returns:
            Model: Derlenmiş model
        """
        # Input
        inputs = Input(shape=(self.max_len,), name='input')
        
        # Embedding
        x = self._get_embedding_layer()(inputs)
        x = SpatialDropout1D(0.2)(x)
        
        # Farklı kernel boyutlarıyla konvolüsyon
        conv_outputs = []
        for kernel_size in kernel_sizes:
            conv = Conv1D(filters, kernel_size, activation='relu', 
                         padding='same', name=f'conv_{kernel_size}')(x)
            pool = GlobalMaxPooling1D(name=f'pool_{kernel_size}')(conv)
            conv_outputs.append(pool)
        
        # Birleştir
        if len(conv_outputs) > 1:
            concat = Concatenate()(conv_outputs)
        else:
            concat = conv_outputs[0]
        
        # Fully connected layers
        dense = Dense(128, activation='relu')(concat)
        dense = Dropout(dropout)(dense)
        dense = Dense(64, activation='relu')(dense)
        dense = Dropout(dropout/2)(dense)
        outputs = Dense(1, activation='sigmoid')(dense)
        
        model = Model(inputs=inputs, outputs=outputs, name='TextCNN_Model')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_lstm(self, filters: int = 64, kernel_size: int = 3,
                       lstm_units: int = 64, dropout: float = 0.5) -> Sequential:
        """
        CNN + LSTM hibrit modeli oluşturur.
        
        CNN yerel özellikleri yakalar, LSTM ise sıralı bağımlılıkları öğrenir.
        
        Args:
            filters: CNN filtre sayısı
            kernel_size: CNN filtre boyutu
            lstm_units: LSTM birim sayısı
            dropout: Dropout oranı
            
        Returns:
            Sequential: Derlenmiş model
        """
        model = Sequential([
            self._get_embedding_layer(),
            SpatialDropout1D(0.2),
            Conv1D(filters, kernel_size, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters * 2, kernel_size, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            LSTM(lstm_units),
            Dropout(dropout),
            Dense(64, activation='relu'),
            Dropout(dropout/2),
            Dense(1, activation='sigmoid')
        ], name='CNN_LSTM_Model')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_bilstm_attention(self, units: int = 64, dropout: float = 0.5) -> Model:
        """
        Attention mekanizmalı BiLSTM modeli oluşturur.
        
        Attention, modelin önemli kelimelere odaklanmasını sağlar.
        
        Args:
            units: LSTM birim sayısı
            dropout: Dropout oranı
            
        Returns:
            Model: Derlenmiş model
        """
        # Input
        inputs = Input(shape=(self.max_len,), name='input')
        
        # Embedding
        x = self._get_embedding_layer()(inputs)
        x = SpatialDropout1D(0.2)(x)
        
        # BiLSTM
        lstm_out = Bidirectional(LSTM(units, return_sequences=True))(x)
        lstm_out = Dropout(dropout)(lstm_out)
        
        # Self-Attention
        attention_output = Attention()([lstm_out, lstm_out])
        
        # Global Average Pooling
        pooled = GlobalAveragePooling1D()(attention_output)
        
        # Fully connected
        dense = Dense(64, activation='relu')(pooled)
        dense = Dropout(dropout/2)(dense)
        outputs = Dense(1, activation='sigmoid')(dense)
        
        model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_Attention_Model')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_deep_lstm(self, units: list = [128, 64, 32], dropout: float = 0.5) -> Sequential:
        """
        Derin (çok katmanlı) LSTM modeli oluşturur.
        
        Args:
            units: Her LSTM katmanı için birim sayıları listesi
            dropout: Dropout oranı
            
        Returns:
            Sequential: Derlenmiş model
        """
        layers = [
            self._get_embedding_layer(),
            SpatialDropout1D(0.2)
        ]
        
        # LSTM katmanları
        for i, unit in enumerate(units[:-1]):
            layers.append(LSTM(unit, return_sequences=True, name=f'lstm_{i+1}'))
            layers.append(Dropout(dropout))
        
        # Son LSTM katmanı
        layers.append(LSTM(units[-1], return_sequences=False, name=f'lstm_{len(units)}'))
        
        # Dense katmanlar
        layers.extend([
            Dense(64, activation='relu'),
            Dropout(dropout/2),
            Dense(1, activation='sigmoid')
        ])
        
        model = Sequential(layers, name='Deep_LSTM_Model')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_ensemble_model(self, dropout: float = 0.5) -> Model:
        """
        Ensemble modeli oluşturur (CNN + BiLSTM birleşimi).
        
        İki farklı mimarinin çıktılarını birleştirir.
        
        Args:
            dropout: Dropout oranı
            
        Returns:
            Model: Derlenmiş model
        """
        # Input
        inputs = Input(shape=(self.max_len,), name='input')
        
        # Embedding (paylaşımlı)
        embedding = self._get_embedding_layer()(inputs)
        embedding = SpatialDropout1D(0.2)(embedding)
        
        # CNN Branch
        cnn = Conv1D(64, 3, activation='relu', padding='same')(embedding)
        cnn = GlobalMaxPooling1D()(cnn)
        
        # BiLSTM Branch
        lstm = Bidirectional(LSTM(64, return_sequences=False))(embedding)
        
        # Birleştir
        concat = Concatenate()([cnn, lstm])
        
        # Fully connected
        dense = Dense(128, activation='relu')(concat)
        dense = Dropout(dropout)(dense)
        dense = Dense(64, activation='relu')(dense)
        dense = Dropout(dropout/2)(dense)
        outputs = Dense(1, activation='sigmoid')(dense)
        
        model = Model(inputs=inputs, outputs=outputs, name='Ensemble_Model')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def get_callbacks(model_path: str, patience: int = 5) -> list:
        """
        Eğitim için callback'leri döndürür.
        
        Args:
            model_path: Model kayıt yolu
            patience: Early stopping için sabır değeri
            
        Returns:
            list: Callback listesi
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        return callbacks
    
    def get_model_summary(self, model_name: str = 'lstm') -> None:
        """
        Belirtilen modelin özetini yazdırır.
        
        Args:
            model_name: Model adı
        """
        model_builders = {
            'simple_rnn': self.build_simple_rnn,
            'lstm': self.build_lstm,
            'bilstm': self.build_bilstm,
            'gru': self.build_gru,
            'cnn': self.build_cnn,
            'cnn_lstm': self.build_cnn_lstm,
            'bilstm_attention': self.build_bilstm_attention,
            'deep_lstm': self.build_deep_lstm,
            'ensemble': self.build_ensemble_model
        }
        
        if model_name not in model_builders:
            print(f" Geçersiz model adı: {model_name}")
            print(f"   Geçerli modeller: {list(model_builders.keys())}")
            return
            
        model = model_builders[model_name]()
        model.summary()
    
    def list_available_models(self) -> list:
        """
        Mevcut model isimlerini listeler.
        
        Returns:
            list: Model isimleri
        """
        models = [
            'simple_rnn',
            'lstm', 
            'bilstm',
            'gru',
            'cnn',
            'cnn_lstm',
            'bilstm_attention',
            'deep_lstm',
            'ensemble'
        ]
        
        print(" Mevcut Modeller:")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model}")
            
        return models


# Test kodu
if __name__ == "__main__":
    # Test parametreleri
    vocab_size = 10000
    embedding_dim = 128
    max_len = 200
    
    # Model oluşturucu
    model_builder = SentimentModels(vocab_size, embedding_dim, max_len)
    
    print("="*60)
    print(" MODEL MİMARİLERİ TESTİ")
    print("="*60)
    
    # Mevcut modelleri listele
    model_builder.list_available_models()
    
    # LSTM modelini göster
    print("\n LSTM Model Özeti:")
    print("-"*40)
    model_builder.get_model_summary('lstm')
    
    print("\n Model testleri başarılı!")

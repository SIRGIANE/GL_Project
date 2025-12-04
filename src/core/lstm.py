"""
LSTM Model for sequence data
"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import time
from .model import BaseModel


class LSTMModel(BaseModel):
    """Modèle LSTM pour données séquentielles"""
    
    def __init__(self, units=100, **kwargs):
        """
        Initialise le modèle LSTM
        
        Args:
            units: Nombre d'unités LSTM
            **kwargs: Paramètres supplémentaires
        """
        super().__init__("LSTM", units=units, **kwargs)
        self.build_model()
    
    def build_model(self):
        """Construit le modèle LSTM"""
        self.model = Sequential()
        
        # Couche LSTM
        self.model.add(LSTM(
            units=self.params['units'],
            return_sequences=True,
            input_shape=(1, self.params.get('input_dim', None))
        ))
        
        # Dropout
        if self.params.get('dropout_rate', 0.2) > 0:
            self.model.add(Dropout(self.params.get('dropout_rate', 0.2)))
        
        # Seconde couche LSTM
        self.model.add(LSTM(
            units=self.params['units'] // 2,
            return_sequences=False
        ))
        
        # Dropout
        if self.params.get('dropout_rate', 0.2) > 0:
            self.model.add(Dropout(self.params.get('dropout_rate', 0.2)))
        
        # Couche de sortie
        self.model.add(Dense(1, activation='sigmoid'))
        
        # Compiler
        self.model.compile(
            optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.001)),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, **kwargs):
        """
        Entraîne le modèle LSTM
        
        Args:
            X_train: Features d'entraînement (reshaped pour LSTM)
            y_train: Labels d'entraînement
            epochs: Nombre d'époques
            batch_size: Taille du batch
            **kwargs: Arguments supplémentaires
        """
        start_time = time.time()
        
        # Reshape pour LSTM: (samples, timesteps, features)
        if len(X_train.shape) == 2:
            X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        else:
            X_train_reshaped = X_train
        
        print(f"🔧 Entraînement du modèle LSTM (units={self.params['units']})...")
        
        history = self.model.fit(
            X_train_reshaped, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=self.params.get('validation_split', 0.2),
            verbose=self.params.get('verbose', 0)
        )
        
        self.training_time = time.time() - start_time
        self.trained = True
        self.history = history.history
        
        print(f"✅ LSTM entraîné en {self.training_time:.2f}s")
        return self
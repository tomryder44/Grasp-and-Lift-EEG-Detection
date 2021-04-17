
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam

def compile_LSTM(time_steps):
    ''' Compiles an LSTM network with given parameters.'''
    
    model = Sequential()
    
    model.add(LSTM(100, input_shape=(time_steps, 22), 
                   return_sequences=False))
    model.add(Dropout(0.2))
    # output layer
    model.add(Dense(6, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=[AUC(name='auc')])
    
    print(model.summary())
    
    return model

    
    
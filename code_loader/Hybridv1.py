# Add necessary imports
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Input, Reshape, LSTM, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

def create_model(optimizer='adam', init_mode='he_uniform', l2_rate=0.01, learning_rate=0.0001):
    base_model = DenseNet121(include_top=False, 
                             weights='imagenet', 
                             input_tensor=Input(shape=(224, 224, 3)))
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Adding LSTM layers
    x = Reshape((1, x.shape[1]))(x)  # Reshape to be compatible with LSTM
    x = LSTM(256, return_sequences=True)(x)
    x = Flatten()(x)
    
    x = Dense(512, activation='relu', kernel_initializer=init_mode, kernel_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)  # Increased dropout rate
    predictions = Dense(21, activation='softmax', kernel_regularizer=l2(l2_rate))(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers[:-30]:  # Fine-tune more layers
        layer.trainable = False
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# Compile and train the model as before
model = create_model()
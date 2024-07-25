# Add necessary imports
from keras.applications.xception import Xception
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

def create_model(input_shape=(224, 224, 3), num_classes=21, l2_rate=0.01):
    # Create the base pre-trained model
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    predictions = Dense(21, activation='softmax')(x)

    # Construct the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create the Xception model
model = create_model()
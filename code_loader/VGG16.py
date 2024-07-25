# Add necessary imports
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

def create_model(input_shape=(224, 224, 3), num_classes=21, l2_rate=0.01):
    # Create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Add custom layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_rate))(x)

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

model = create_model()
# Add necessary imports
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

def create_model(optimizer='adam', init_mode='uniform', l2_rate=0.01):
    # Load an InceptionV3 model pre-trained on ImageNet
    base_model = InceptionV3(include_top=False, 
                             weights='imagenet', 
                             input_tensor=Input(shape=(224, 224, 3)))

    # Adding custom layers on top of InceptionV3
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(21, activation='softmax')(x)
    
    # Construct the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Freeze the layers of the base model to not train them again
    for layer in base_model.layers:
        layer.trainable = False

    return model

# Replace the model creation line with the new InceptionV3 model
model = create_model()
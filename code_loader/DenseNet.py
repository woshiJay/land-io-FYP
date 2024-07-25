# Add necessary imports
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Input
from keras.models import Model
from keras.regularizers import l2

def create_model(optimizer='adam', init_mode='uniform', l2_rate=0.01):
    # Load a DenseNet model pre-trained on ImageNet
    base_model = DenseNet121(include_top=False, 
                             weights='imagenet', 
                             input_tensor=Input(shape=(224, 224, 3)))

    # Adding custom layers on top of DenseNet
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

# Replace the model creation line with the new DenseNet model
model = create_model()
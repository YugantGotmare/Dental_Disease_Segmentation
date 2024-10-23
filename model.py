import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define the U-Net model with VGG16 as a backbone
def unet_vgg16_model(input_shape):
    """
    Builds a U-Net model using the VGG16 architecture as a backbone.
    
    Parameters:
    input_shape (tuple): Shape of the input image (height, width, channels).
    
    Returns:
    Model: Compiled U-Net model.
    """
    # Load VGG16 with ImageNet weights, excluding the top layer
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze VGG16 layers to prevent them from training
    for layer in vgg_base.layers:
        layer.trainable = False

    inputs = vgg_base.input
    c1 = vgg_base.get_layer('block1_conv2').output
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = vgg_base.get_layer('block2_conv2').output
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = vgg_base.get_layer('block3_conv3').output
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = vgg_base.get_layer('block4_conv3').output

    # Decoder
    u5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.5)(c5)

    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)

    # Output layer: 1 channel for binary mask prediction
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    # Create the U-Net model
    model = Model(inputs, outputs)

    # Compile the model with Adam optimizer and binary cross-entropy loss
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

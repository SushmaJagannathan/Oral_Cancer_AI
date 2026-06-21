import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np
import os


IMG_SIZE=(224,224)
BATCH=16



data=ImageDataGenerator(

    preprocessing_function=
    tf.keras.applications.efficientnet.preprocess_input,

    validation_split=0.2,

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.3,

    shear_range=0.2,

    brightness_range=[0.7,1.3],

    horizontal_flip=True,

    fill_mode="nearest"

)



train=data.flow_from_directory(

    "dataset",

    target_size=IMG_SIZE,

    batch_size=BATCH,

    class_mode="binary",

    subset="training"

)



val=data.flow_from_directory(

    "dataset",

    target_size=IMG_SIZE,

    batch_size=BATCH,

    class_mode="binary",

    subset="validation",

    shuffle=False

)



print(
    "CLASS ORDER:",
    train.class_indices
)



weights=class_weight.compute_class_weight(

    class_weight="balanced",

    classes=np.unique(train.classes),

    y=train.classes

)


weights=dict(enumerate(weights))


print(
    "CLASS WEIGHTS:",
    weights
)




base=tf.keras.applications.EfficientNetB0(

    include_top=False,

    weights="imagenet",

    input_shape=(224,224,3),

    pooling="avg"

)



base.trainable=False



model=tf.keras.Sequential([

    base,

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),


    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),


    tf.keras.layers.Dropout(0.4),


    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )

])




model.compile(

    optimizer=tf.keras.optimizers.Adam(1e-4),

    loss="binary_crossentropy",

    metrics=["accuracy"]

)



model.fit(

    train,

    validation_data=val,

    epochs=10,

    class_weight=weights

)




# fine tuning

base.trainable=True


for layer in base.layers[:-30]:

    layer.trainable=False




model.compile(

    optimizer=tf.keras.optimizers.Adam(1e-5),

    loss="binary_crossentropy",

    metrics=["accuracy"]

)



model.fit(

    train,

    validation_data=val,

    epochs=15,

    class_weight=weights

)




os.makedirs(
    "models",
    exist_ok=True
)



model.save(
    "models/model.keras"
)


print("MODEL SAVED")
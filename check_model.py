import tensorflow as tf

model = tf.keras.models.load_model(
    "models/model.keras"
)

print("MODEL LOADED SUCCESSFULLY")

model.summary()
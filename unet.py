import tensorflow as tf
from tensorflow.keras import layers,Model



def conv_block(x,filters):

    x=layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x=layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    return x




def build_unet():

    inputs=layers.Input(
        (224,224,3)
    )


    c1=conv_block(
        inputs,
        32
    )

    p1=layers.MaxPooling2D()(c1)



    c2=conv_block(
        p1,
        64
    )

    p2=layers.MaxPooling2D()(c2)



    c3=conv_block(
        p2,
        128
    )



    u1=layers.UpSampling2D()(c3)


    u1=layers.concatenate(
        [u1,c2]
    )


    c4=conv_block(
        u1,
        64
    )



    u2=layers.UpSampling2D()(c4)


    u2=layers.concatenate(
        [u2,c1]
    )


    c5=conv_block(
        u2,
        32
    )


    output=layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c5)



    return Model(
        inputs,
        output
    )
import tensorflow as tf
import numpy as np
import cv2



def create_gradcam(
    image_path,
    model,
    save_path
):


    img=cv2.imread(
        image_path
    )


    img=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    img=cv2.resize(
        img,
        (224,224)
    )


    x=tf.keras.applications.efficientnet.preprocess_input(
        img
    )


    x=np.expand_dims(
        x,
        0
    )



    base=model.layers[0]



    last_conv=None


    for layer in reversed(base.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            last_conv=layer.name
            break




    grad_model=tf.keras.Model(

        base.input,

        [
        base.get_layer(last_conv).output,
        base.output
        ]

    )



    with tf.GradientTape() as tape:


        conv,pred=grad_model(x)

        loss=pred[:,0]



    grads=tape.gradient(
        loss,
        conv
    )


    weights=tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )



    conv=conv[0]



    heat=conv @ weights[...,None]



    heat=tf.squeeze(
        heat
    )



    heat=np.maximum(
        heat,
        0
    )


    heat/=(
        np.max(heat)+1e-8
    )



    heat=cv2.resize(
    heat,
    (224,224)
    )


    heat=np.uint8(
        255*heat
    )



    heat=cv2.applyColorMap(
        heat,
        cv2.COLORMAP_JET
    )



    original=cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )



    output=cv2.addWeighted(
        original,
        0.6,
        heat,
        0.4,
        0
    )


    cv2.imwrite(
        save_path,
        output
    )
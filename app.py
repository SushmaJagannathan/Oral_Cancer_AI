from flask import Flask,request,render_template,jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import uuid

from flask_cors import CORS

from gradcam import create_gradcam
from scorecam import ScoreCAM




app=Flask(__name__)

CORS(app)



# =====================
# LOAD CLASSIFIER
# =====================

model=tf.keras.models.load_model(
    "models/model.keras"
)




# =====================
# FOLDERS
# =====================

os.makedirs(
    "static/uploads",
    exist_ok=True
)


os.makedirs(
    "static/heatmaps",
    exist_ok=True
)

@app.route("/")
def home():

    return render_template(
        "index.html"
    )





@app.route(
"/predict",
methods=["POST"]
)
def predict():



    file=request.files["image"]



    name=str(uuid.uuid4())+".jpg"


    path="static/uploads/"+name



    file.save(path)





    # =====================
    # PREPROCESS
    # =====================


    img=cv2.imread(path)


    img=cv2.resize(
        img,
        (224,224)
    )


    img=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )



    x=tf.keras.applications.efficientnet.preprocess_input(
        img
    )


    x=np.expand_dims(
        x,
        axis=0
    )





    # =====================
    # PREDICTION
    # =====================


    score=model.predict(x)[0][0]


    print("================")
    print("MODEL SCORE:",score)
    print("================")





    if score < 0.5:


        prediction="Cancer Detected"

        confidence=(1-score)*100

        risk="High"



    else:


        prediction="Normal"

        confidence=score*100

        risk="Low"





    confidence=min(
        confidence,
        99.5
    )





    # =====================
    # GRAD CAM
    # =====================


    grad_path=(
        "static/heatmaps/grad_"
        +name
    )


    create_gradcam(

        path,

        model,

        grad_path

    )





    # =====================
    # SCORE CAM
    # =====================


    scorecam=ScoreCAM(model)



    score_image=scorecam.generate(
        path
    )



    score_path=(
        "static/heatmaps/score_"
        +name
    )



    cv2.imwrite(
        score_path,
        score_image
    )







    return jsonify({


        "prediction":
        prediction,


        "confidence":
        round(
            float(confidence),
            2
        ),


        "risk":
        risk,


        "gradcam":
        "/"+grad_path,


        "scorecam":
        "/"+score_path,



    })





if __name__=="__main__":


    app.run(
        debug=True
    )
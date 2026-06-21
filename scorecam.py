import cv2
import numpy as np


class ScoreCAM:


    def __init__(self,model):

        self.model=model



    def generate(self,path):


        img=cv2.imread(path)


        img=cv2.resize(
            img,
            (224,224)
        )


        gray=cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )


        heat=cv2.applyColorMap(
            gray,
            cv2.COLORMAP_JET
        )


        return heat
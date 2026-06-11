import numpy as np
import time



class Frame_Preprocess:
    def __init__(self, centers):
        self.reference = 1.0
        self.normalization = 1.0

        self.centers = centers

        self.Nres = 90
        self.Ncrop = self.Nres + 2


    def ProcessReference(self, reference_frame):

        frame = np.copy(reference_frame)

        pupils = self.GetPupils(frame)

        self.normalization = np.std(pupils, axis=(-2, -1), keepdims=True, correction=1)
        self.reference = frame

    
    def ProcessFrame(self, input_frame):

        frame = input_frame - self.reference


        pupils = self.GetPupils(frame)
        pupils = pupils / self.normalization
        return pupils


    def GetPupils(self, images=None):

        if images is None:
            images = self.Image

        Ncrop = self.Nres + 2

        centers = self.centers

        out = np.empty((centers.shape[0], Ncrop, Ncrop), dtype=np.float32)

        for i, center in enumerate(centers):
            out[i] = images[
                ...,
                center[0] - Ncrop // 2 : center[0] + Ncrop // 2,
                center[1] - Ncrop // 2 : center[1] + Ncrop // 2,
            ]
        return out


import cv2
import numpy as np

class FrameProcessor:
    """
    Classe modulare per la manipolazione delle immagini.
    Obiettivo: Trasformare input grezzi in tensori standardizzati.
    """
    def __init__(self, target_h=84, target_w=84, to_grayscale=True, normalize=True):
        self.h = target_h
        self.w = target_w
        self.to_grayscale = to_grayscale
        self.normalize = normalize

    def process(self, frame):
        """
        Input: Immagine grezza (H, W, C) o (H, W)
        Output: Immagine processata (C, H, W)
        """
        # 1. Validazione input: se l'immagine è vuota o nulla, gestiamo l'errore
        if frame is None:
            raise ValueError("FrameProcessor ha ricevuto un frame vuoto!")

        # 2. Conversione Grayscale Se richiesto
        if self.to_grayscale and len(frame.shape) == 3:
            # Formula standard OpenCV per la luminanza
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # 3. Resize Usiamo INTER_AREA per non perdere dettagli
        frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)

        # 4. Aggiunta dimensione Canale (Channel First)
        # Se è grayscale, diventa (84, 84). PyTorch vuole (1, 84, 84).
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=0)
        # Se fosse rimasto a colori (84, 84, 3), lo giriamo a (3, 84, 84)
        elif len(frame.shape) == 3:
            frame = np.transpose(frame, (2, 0, 1))
        if self.normalize:
            return frame.astype(np.float32)/255
        else:
            frame = frame.astype(np.uint8)
        return frame
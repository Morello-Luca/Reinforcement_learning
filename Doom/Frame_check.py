import cv2
import numpy as np
from ImgProcess import FrameProcessor
import matplotlib.pyplot as plt
# 1. Carica un'immagine (attenzione: OpenCV carica in BGR)
img = cv2.imread("test.png")
# Se vuoi essere coerente con COLOR_RGB2GRAY, converti prima
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 2. Inizializza il processor
processor = FrameProcessor(target_h=84, target_w=84, to_grayscale=True,normalize=True)
# 3. Processa il frame
processed = processor.process(img)
plt.figure(frameon=False)
plt.imshow(processed[0], cmap="gray")
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
# 4. Controlla forma e tipo
print("Shape:", processed.shape)   # (1, 84, 84)
print("Dtype:", processed.dtype)   # uint8
plt.imshow(processed[0], cmap="gray")
plt.colorbar()
plt.title("Processed Frame (normalized)")
plt.show()




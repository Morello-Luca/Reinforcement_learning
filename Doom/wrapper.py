import gymnasium as gym
from gymnasium import spaces
import numpy as np
# Importiamo il nostro modulo generico
from ImgProcess import FrameProcessor

class VisualAdapter(gym.ObservationWrapper):
    """
    Wrapper universale che collega un ambiente Gym al FrameProcessor.
    """
    def __init__(self, env, target_h=84, target_w=84, grayscale=True,normalize=True):
        super().__init__(env)
        
        # Istanziamo il processore modulare
        self.processor = FrameProcessor(target_h, target_w, grayscale,normalize)
        
        # Calcoliamo i canali finali (1 se grigio, 3 se RGB)
        #c_channels = 1 if grayscale else 3
        
        # Definiamo lo spazio delle osservazioni per l'agente 
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            #shape=(c_channels, target_h, target_w),
            shape=(target_h, target_w), 
            dtype=np.float32
        )

    def observation(self, observation):
        """
        Qui gestiamo le specificità dell'ambiente.
        """
        # Esempio specifico per ViZDoom che a volte mette il frame dentro un dizionario
        if isinstance(observation, dict):
            frame = observation.get("screen", observation) # Cerca la chiave "screen"
        else:
            frame = observation
        processed = self.processor.process(frame)
        # Usiamo il processore generico per fare il lavoro sporco
        return processed.squeeze()
    
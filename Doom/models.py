import torch
import torch.nn as nn
import numpy as np

class VisualEncoder(nn.Module):
    """
    Modulo CNN (Convolutional Neural Network).
    Ruolo: Trasformare i pixel in un vettore di feature compatto.
    """
    def __init__(self, input_shape, output_dim=512):
        super(VisualEncoder, self).__init__()
        
        # input_shape sarà (C, H, W), es: (4, 84, 84)
        c, h, w = input_shape
        
        # Definizione dei 3 strati convoluzionali (Standard NatureCNN)
        self.conv = nn.Sequential(
            # Layer 1: Cerca feature grossolane (bordi)
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Layer 2: Cerca forme più complesse
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Layer 3: Cerca dettagli specifici
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calcolo automatico della dimensione piatta

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy_input)
            self.flatten_dim = conv_out.view(1, -1).size(1)
            
        # Layer finale che porta tutto alla dimensione richiesta (es. 512 feature)
        self.fc = nn.Linear(self.flatten_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Passaggio attraverso i filtri convoluzionali
        x = self.conv(x)
        # Appiattimento: da (Batch, 64, 7, 7) a (Batch, 3136)
        x = x.view(x.size(0), -1)
        # Compressione finale a 512 feature
        x = self.relu(self.fc(x))
        return x
    


class DQNHead(nn.Module):
    """
    Modulo Controller (MLP).
    Ruolo: Mappare le feature visive in Q-values per ogni azione.
    """
    def __init__(self, input_dim=512, n_actions=5):
        super(DQNHead, self).__init__()
        
        self.net = nn.Sequential(
            # Strato nascosto per ragionare sulle feature
            nn.Linear(input_dim, 512),
            nn.ReLU(),          
            # Strato di output: N neuroni pari al numero di azioni possibili
            # Nota: Non usiamo attivazioni qui (niente ReLU/Sigmoid) perché  i Q-values possono essere negativi o positivi senza limiti.
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)
    
class DoomAgent(nn.Module):
    """
    Assemblaggio finale.
    """
    def __init__(self, input_shape, n_actions):
        super(DoomAgent, self).__init__()
        # 1. Montiamo gli occhi (CNN)
        self.encoder = VisualEncoder(input_shape=input_shape)
        # 2. Montiamo il cervello (DQN)
        # Nota: L'encoder produce 512 feature di default, quindi la head deve aspettarsene 512
        self.head = DQNHead(input_dim=512, n_actions=n_actions)

    def forward(self, x):
        # Flusso dati: Pixel -> Encoder -> Features -> Head -> Q-Values
        features = self.encoder(x)
        q_values = self.head(features)
        return q_values
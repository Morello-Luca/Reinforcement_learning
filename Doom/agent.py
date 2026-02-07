import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from models import DoomAgent  

class ReplayBuffer:
    """
    La memoria dell'agente.
    Salva le transizioni (Stato, Azione, Reward, Nuovo Stato, Done).
    """
    
    def __init__(self, capacity=10000):
        # Usiamo una deque: quando è piena, elimina automaticamente i ricordi più vecchi.
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Aggiunge un'esperienza alla memoria."""
        # Nota: state e next_state sono già i tensori processati (4, 84, 84)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Estrae un 'batch' casuale di ricordi.
        Perché casuale? Per rompere la correlazione tra frame consecutivi 
        (altrimenti l'IA si fisserebbe troppo su quello che è appena successo).
        """
        # Estraiamo N esperienze a caso
        batch = random.sample(self.buffer, batch_size)
        
        # Separiamo i dati in gruppi (tutti gli stati insieme, tutte le azioni insieme...)
        # e li trasformiamo in array NumPy pronti per PyTorch.
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_shape, n_actions, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=10000):
        self.n_actions = n_actions
        self.gamma = gamma  # Fattore di sconto (quanto conta il futuro rispetto all'oggi)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Iperparametri per Epsilon-Greedy ---
        # All'inizio l'agente è curioso (Epsilon alto) e fa azioni a caso.
        # Poi diventa esperto (Epsilon basso) e usa la rete neurale.
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # --- Creazione dei Cervelli ---
        # 1. Policy Net: quella che alleniamo
        self.policy_net = DoomAgent(state_shape, n_actions).to(self.device)
        # 2. Target Net: quella stabile per calcolare l'errore
        self.target_net = DoomAgent(state_shape, n_actions).to(self.device)
        # Allineiamo subito i pesi della target con la policy
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # La target net non deve mai andare in training mode

        # Ottimizzatore: Lo strumento che aggiusta i pesi (Adam è lo standard)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        """
        Sceglie l'azione usando la strategia Epsilon-Greedy.
        Input: state (numpy array 4x84x84)
        """
        # Calcolo la soglia di esplorazione attuale
        self.steps_done += 1
        # Formula di decadimento lineare (puoi usarne anche esponenziali)
        self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay)

        # Tiro un dado: se esce un numero basso, esploro (azione casuale)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        # Altrimenti sfrutto la conoscenza (chiedo alla rete)
        with torch.no_grad():
            # Trasformo l'input in tensore PyTorch e aggiungo la dimensione batch (da 4x84x84 a 1x4x84x84)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            # Prendo l'indice dell'azione con il valore più alto
            return q_values.argmax().item()

    def learn(self, memory, batch_size=32):
        """
        Il cuore dell'addestramento: aggiornamento pesi tramite Backpropagation.
        """
        # Se non ho abbastanza ricordi, non faccio nulla
        if len(memory) < batch_size:
            return

        # 1. Recupero un batch di esperienze dalla memoria
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # Converto tutto in tensori GPU/CPU
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device) # unsqueeze serve per le operazioni matriciali
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 2. CALCOLO IL VALORE ATTUALE (Predetto dalla Policy Net)
        # "Quanto pensavo valesse l'azione che ho fatto in quello stato?"
        # gather(1, actions) seleziona solo il Q-value dell'azione effettivamente presa
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # 3. CALCOLO IL VALORE ATTESO (Obiettivo dato dalla Target Net + Ricompensa reale)
        # "Quanto valeva davvero? (Ricompensa immediata + valore del miglior stato futuro)"
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            # Se done=1 (gioco finito), non c'è futuro, quindi il valore è solo la reward
            expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 4. CALCOLO LA PERDITA (LOSS)
        # Errore Quadratico Medio (MSE) o Huber Loss (SmoothL1) tra predizione e realtà
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # 5. AGGIORNAMENTO PESI (Backpropagation)
        self.optimizer.zero_grad() # Resetta i gradienti vecchi
        loss.backward()            # Calcola i nuovi gradienti (quanto ogni peso ha contribuito all'errore)
        
        # Gradient Clipping: evita che i gradienti esplodano (molto utile con le CNN)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()      # Applica le modifiche ai pesi

    def update_target_network(self):
        """Copia i pesi dalla Policy alla Target. Da chiamare ogni N step."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
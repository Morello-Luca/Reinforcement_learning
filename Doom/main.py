import gymnasium as gym
import vizdoom
from vizdoom import gymnasium_wrapper # Prova questo import specifico per la 1.2.4
from gymnasium.wrappers import FrameStackObservation
# Se il comando sopra fallisce ancora, usa questo trucco:
import sys
# Forza l'ambiente a trovare le risorse di vizdoom
from gymnasium.envs.registration import register
import torch
import numpy as np
import matplotlib.pyplot as plt

# Importiamo i moduli che abbiamo costruito
from wrapper import VisualAdapter
from agent import DQNAgent, ReplayBuffer

from manager import ModelHandler


# --- 1. IPERPARAMETRI (Il pannello di controllo) ---
BATCH_SIZE = 64         # Quante esperienze l'IA ripassa ogni volta [cite: 102]
GAMMA = 0.95            # Quanto conta il futuro (0.99 = molto lungimirante)
EPS_START = 1.0         # Curiosità iniziale (100% casuale)
EPS_END = 0.02          # Curiosità finale (2% casuale)
EPS_DECAY = 100000       # Quanto velocemente smette di esplorare
LR = 5e-5               # Learning Rate: velocità di apprendimento
TARGET_UPDATE = 4000    # Ogni quanti step aggiorniamo la "Target Network"
MEMORY_SIZE = 200000     # Dimensione del diario dei ricordi
NUM_EPISODES = 2500      # Quante partite far giocare
MAX_STEPS = 1000        # Durata massima di una partita (per evitare loop infiniti)

def make_env(scenario="VizdoomBasic-v0", render_mode=None):
    """
    Funzione Factory per assemblare l'ambiente con i pezzi Lego.
    """
    # 1. Crea l'ambiente base (assumiamo che gym-vizdoom sia installato)
    # Nota: Se usi la libreria vizdoom diretta, il codice di init cambia leggermente,
    # ma la logica dei wrapper resta identica.
    env = gym.make(scenario, render_mode=render_mode)
    
    # 2. Applica gli 'occhiali' (VisualAdapter)
    # Ridimensiona a 84x84 e converte in scala di grigi [cite: 91, 130]
    env = VisualAdapter(env, target_h=84, target_w=84, grayscale=True)
    
    # 3. Applica la memoria temporale (FrameStack)
    # Impila 4 frame consecutivi per dare il senso del movimento [cite: 65, 66]
    # Output finale: Tensore (4, 84, 84)
    env = FrameStackObservation(env, stack_size=4)
    
    return env

def main():
    # --- 2. INIZIALIZZAZIONE ---
    print("Inizializzazione Ambiente...")
    env = make_env()
    
    # Recuperiamo le dimensioni automatiche
    # shape sarà (4, 84, 84) come definito nel progetto [cite: 63]
    state_shape = env.observation_space.shape 
    n_actions = int(env.action_space.n)

    if n_actions > 0:
        print(f"Azioni disponibili: {n_actions}")
        
    
    print(f"Stato: {state_shape}, Azioni: {n_actions}")
    print("Inizializzazione Agente...")
    
    # Creiamo il cervello e la memoria
    agent = DQNAgent(state_shape, n_actions, lr=LR, gamma=GAMMA, 
                     epsilon_start=EPS_START, epsilon_end=EPS_END, 
                     epsilon_decay=EPS_DECAY)
    
    memory = ReplayBuffer(capacity=MEMORY_SIZE)
    
    rewards_history = [] # Per tracciare i progressi
    global_step_count = 0
    handler = ModelHandler()
    best_avg_reward = -float('inf')
    # --- 3. TRAINING LOOP  ---
    for episode in range(NUM_EPISODES):
        
        # Reset dell'ambiente all'inizio della partita
        # Gym restituisce (osservazione, info)
        state, _ = env.reset()
        
        # In Gym il FrameStack restituisce un LazyFrame, lo convertiamo in array numpy
        state = np.array(state) 
        
        total_reward = 0
        done = False
        
        for step in range(MAX_STEPS):
            global_step_count += 1
            
            # A. SCEGLIERE L'AZIONE
            # L'agente decide se esplorare o sfruttare
            action = agent.select_action(state)
            
            # B. ESEGUIRE L'AZIONE NELL'AMBIENTE
            # step() restituisce 5 valori:
            # next_state: la nuova immagine (4, 84, 84)
            # reward: punti ottenuti (+1, -0.1, etc) [cite: 79, 80]
            # terminated: gioco finito (vittoria/morte)
            # truncated: tempo scaduto
            # info: dati extra
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Uniamo i due tipi di "fine gioco"
            done = terminated or truncated
            next_state = np.array(next_state)

            # C. MEMORIZZARE L'ESPERIENZA
            memory.push(state, action, reward, next_state, done)
            
            # Avanziamo di stato
            state = next_state
            total_reward += reward
            
            # D. IMPARARE (Training Step)
            # Aggiornamento pesi della CNN e del MLP 
            agent.learn(memory, BATCH_SIZE)
            
            # E. AGGIORNARE LA TARGET NETWORK
            # Ogni tanto, stabilizziamo la conoscenza
            if global_step_count % TARGET_UPDATE == 0:
                agent.update_target_network()
            
            if done:
                break
        
        # --- LOGGING ---
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-60:]) # Media ultime 60 partite
        if avg_reward > best_avg_reward and episode > 50: # Aspetta che l'agente impari un po'
            best_avg_reward = avg_reward
            handler.save_best(agent, avg_reward)
        
        print(f"Episodio {episode}/{NUM_EPISODES} | "
              f"Reward: {total_reward:.2f} | "
              f"Avg 10: {avg_reward:.2f} | "
              f"Epsilon: {agent.epsilon:.4f}")

    print("Addestramento completato!")
    env.close()
    
    # Plot finale per vedere se ha imparato
    plt.plot(rewards_history)
    plt.title("Curva di Apprendimento Doom")
    plt.xlabel("Episodi")
    plt.ylabel("Reward Totale")
    plt.show()

if __name__ == "__main__":
    main()
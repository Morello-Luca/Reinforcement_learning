import time
import torch
import numpy as np

# Importiamo i tuoi moduli
from main import make_env  
from agent import DQNAgent
from manager import ModelHandler

def watch_agent():
    print("\n--- Apertura Finestra Doom: Enjoy Mode ---")
    
    # 1. Setup Ambiente FORZANDO il render human
    # Passiamo render_mode="human" qui, così il main.py resta veloce (None)
    env = make_env(render_mode="human") 
    
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # 2. Setup Agente (Epsilon 0 = nessuna mossa a caso)
    agent = DQNAgent(state_shape, n_actions, epsilon_start=0.0, epsilon_end=0.0)
    
    # 3. Caricamento pesi
    handler = ModelHandler()
    if not handler.load(agent):
        print("ERRORE: Modello non trovato. Controlla la cartella 'checkpoints'.")
        return

    # 4. Loop di gioco visivo
    for episode in range(5):
        state, _ = env.reset()
        state = np.array(state) # Formato (4, 84, 84)
        
        done = False
        total_reward = 0
        print(f"Giocando Episodio {episode+1}...")

        while not done:
            # L'agente decide in base ai pixel che vede
            action = agent.select_action(state)
            
            # Applichiamo l'azione in Doom
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # AGGIORNAMENTO STATO: fondamentale usare next_state
            state = np.array(next_state)
            
            done = terminated or truncated
            total_reward += reward
            
            # Delay per vedere l'azione (50 FPS circa)
            time.sleep(0.02) 

        print(f"Episodio {episode+1} terminato | Reward: {total_reward:.2f}")
    
    env.close()
    print("Sessione conclusa.")

if __name__ == "__main__":
    watch_agent()
import torch
import os

class ModelHandler:
    def __init__(self, save_dir="checkpoints"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save_best(self, agent, reward):
        """Salva il modello come il migliore trovato finora."""
        path = os.path.join(self.save_dir, "best_doom_model.pth")
        torch.save(agent.policy_net.state_dict(), path)
        print(f"--- Modello salvato con reward: {reward:.2f} ---")

    def load(self, agent, filename="best_doom_model.pth"):
        """Carica i pesi nel cervello dell'agente."""
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            agent.policy_net.load_state_dict(torch.load(path, map_location=agent.device))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"--- Modello {filename} caricato correttamente! ---")
            return True
        print("--- Nessun salvataggio trovato. ---")
        return False
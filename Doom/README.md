# RL - project

Last edited time: February 7, 2026 9:20 PM
Status: Not started

# 1. Overview: Visual Reinforcement Learning Agent

---

## 1  Objective

The project involves the development of a **reinforcement learning system integrated with a convolutional neural network**. 
The system uses **raw visual data**, represented as pixel matrices, as input states, allowing the agent to learn an effective representation of the environment directly from images, without relying on a manual feature-extraction phase.

---

# 2. Methodology and Procedure

The adopted methodology is **Deep Reinforcement Learning (DRL)**, with a **model-free approach** based on convolutional neural networks.

## 2.1 Definition of the State (State Space)

The state **is not a single image**, but a temporal representation of the environment. Specifically:

- **Data type:** Four-dimensional tensor $(N, C, H, W)$, representing both the spatial dimensions of the images and the temporal sequence of frames.
- **Visual state:** Resized images (es. $84 \times 84$ pixel) to standardize the input and reduce computational load..
- **Temporality (Frame Stacking):** The state consists of k consecutive frames (typically $k=4$) to capture **velocity, direction, and temporal dynamics,** addressing the problem of **partial observability**.
- **Preprocessing:** Conversion to **grayscale** to reduce complexity, and **normalization**. $[0, 1]$ and **cropping of non-informative areas** (e.g., HUD or image borders).

## 2.2 **Action Space**

- Turn Left
- Turn Right
- Shoot
- Nothing

## 2.3 Reward Function

The reward function is internal to the environment, so no explicit reward is provided.

---

# 3. RL Architecture

- **Encoder CNN (Convolutional Neural Network):** Three or more convolutional layers to compress the pixels into **a feature vector**.
- **Controller MLP (Multi-Layer Perceptron)**: Receives the feature vector and estimates the action with the highest expected value.

---

# 4. **Brief Roadmap**

- **Environment:** Set up **ViZDoom** to generate images and provide the reward function.
- **Wrapper:** Python class that captures frames, converts them to **grayscale**, resizes them, stacks them, and organizes them into a **tensor**.
- **Agent:** Uses the models to estimate the action according to the defined policy.
- **Training:** Observation → action → reward → **CNN weights update** cycle.
- **Save:** Saving **model checkpoints**.

![image.png](image/image.png)

---

# 5. Pipeline

## 5.1 Image Processing

The files dedicated to **image processing** contain the functions necessary to transform the frames captured from the environment into inputs usable by the neural network. In particular, they leverage the **OpenCV library** to perform the operations described in the previous section, converting the images into a **standardized representation** compatible with the deep learning model.

It is possible to visualize the result by **creating an instance of the class** and providing it with a **test image** as input. The output can then be **plotted to verify correctness**: the network will see an **$84 × 84$ matrix** of values **normalized in the range $[0, 1]$**.

Input image

![2.png](image/2.png)

output image

![test_grayscale1.png](image/test_greyscale1.png)

![test_grayscale.png](image/test_greyscale.png)

## 5.2 Wrapper

The wrapper acts as a **bridge between the image processor and the specific environment** (in this case, ViZDoom). If in the future a different environment, such as Mario, were to be used, **only the wrapper would need to be modified**, while the **`FrameProcessor` would remain unchanged**.

The current code processes a **single preprocessed frame** (dimensions **$1 × 84 × 84$**); to capture the **temporal dynamics** of the environment, the process needs to be **extended to handle sequences of consecutive frames**.

Instead of implementing custom code, we will use the **standard Gym wrapper**:

```python
gym.wrappers.FrameStack
```

This wrapper directly generates the **final tensor** with dimensions **$(4, 84, 84)$**. Essentially, it is equivalent to **stacking four matrices into a single tensor**, or visually, **creating a “cube” composed of four slices along the temporal dimension**.

## 5.3 Network Backbone

We will build the architecture in a **modular way**, separating the **Encoder (CNN)**, responsible for extracting features from the frames, from the **Controller (DQN)**, which uses these features to estimate actions according to the agent’s policy. The two models will be implemented in **separate classes**, and a **final class** will handle instantiating them and connecting them together.

### 5.3.1 Encoder

This module receives as input a tensor with dimensions **$(Batch, 4, 84, 84)$** and compresses it into a **feature vector**.

**Educational concept:** Imagine the pixels as thousands of tiny dots. The **`Conv2d` layers (convolutional layers)** look for patterns:

- **First layer** detects edges and lines.
- **Second layer** recognizes more complex shapes, like circles and corners.
- Th**ird layer** identifies objects or structured combinations of shapes.

At the end, the output is **flattened** into a numerical vector that summarizes all the relevant information.

**Key parameters of a convolutional layer:**

- **in_channels:** number of input channels.
- **out_channels:** number of output channels, i.e., how many filters the layer applies.
- **kernel_size:** size of the convolutional filter, which scans the image in blocks of `kernel_size × kernel_size` pixels.
- **stride:** step size of the filter as it moves across the image.

**Example:**

```python
nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
```

**In this case:**

- The input has **32 channels**.
- The layer applies **64 filters** of size **4×4**.
- The filter moves **2 pixels at a time**.
- The output consists of **64 feature maps**, ready for further processing.

### 5.3.2 La Head (DQN)

This module takes the **summary produced by the CNN** and decides which action to take.

**Educational concept:** This is a classic **fully connected (dense) neural network**. It receives as input a **512-dimensional vector** representing the scene (for example, “there is an enemy on the right”) and calculates a **Q-value** for each possible controller command.

The **selected action** is the one with the **highest Q-value**: for instance, if the value associated with “Shoot” is the highest, the agent executes the **shooting action**.

### 5.3.3 Assembly

At this stage, we **combine the modules**. The modular approach allows the **Controller to be easily replaced**: for example, the **`DQNHead`** can be detached and a **`PPOHead`** connected in the future, without needing to modify the **Encoder (CNN)**.

### 5.4 Agent

This class represents the **decision-making core of the agent** and implements the **Deep Q-Learning** mechanism. Within it, actions are computed and the parameters that guide the agent’s behavior are updated.

A key element of the **DQN algorithm** is the use of **two separate but structurally identical neural networks**, each serving a different purpose:

- **Policy Network:** The active network, used to select actions and continuously optimized during the learning phase.
- **Target Network:** A temporarily fixed version of the network, used to estimate target values during updates. Its use reduces training instabilities that would arise if the network were updated using its own constantly changing estimates as references.

At regular intervals, the parameters of the **Policy Network** are copied to the **Target Network**, updating the reference model while preserving the stability of the learning process.

## 5.5 Main

This is the **final stage of the system**. The `main.py` file acts as the **central coordinator**, orchestrating all the developed components: the **simulation environment** (ViZDoom), the **visual perception module** (Wrapper), the **experience memory** (Replay Buffer), and the agent’s **decision-making module** (Agent).

**1. `make_env`**

In this function, the project’s **modular philosophy** is applied. The process starts from a **raw environment**, to which the **VisualAdapter** (responsible for preprocessing operations such as resizing and grayscale conversion) and the **FrameStack**, which introduces temporal memory, are progressively added.

The final result is an environment that provides the agent exclusively with **preprocessed tensors of size (4, 84, 84)**, ensuring a **clean and uniform input**.

**2. Training loop: `for step in range(MAX_STEPS)`**

This loop represents the **operational core of training**. At each iteration, the following fundamental phases occur:

- the agent observes the current state in the form of pixels,
- selects and executes an action,
- the environment returns the new state and the reward,
- the experience is stored in the buffer,
- the neural network weights are updated via **backpropagation**.

## 5.6 save and replay

In **Reinforcement Learning**, it is crucial to save the **Best Model**, i.e., the model that achieves the highest score during training. The training process can be unstable: performance may fluctuate, and an effective model may degrade in the later stages of learning. Saving the best configuration therefore makes it possible to preserve a **winning policy**, regardless of how training evolves afterward.

For this reason, it is useful to design a **dedicated model-saving system** and an **evaluation script** (“Cinema”) that allows the agent to be observed in action using the best model.

To this end, a new file called **`manager.py`** is created. This class is responsible for:

- initializing and managing the **save directory**,
- saving the model weights in **PyTorch `.pth` format**,
- keeping track of the **best-performing model**.

Create a file called **`enjoy.py`**. This script will:

- **Load the previously saved model**,
- **Completely disable exploration** by setting **ϵ = 0**,
- Launch the environment in **full-screen rendering mode**, allowing you to **observe the agent’s behavior during execution**.

It essentially turns the system into a **playback/evaluation mode**, where the agent acts purely according to the learned policy without random actions.

# 6. FootNote

## 1.1 Image Processing

### 1.1 Il Tensore Quadridimensionale $(N, C, H, W)$

Quando catturi un'immagine, hai dei dati grezzi. Per darli in pasto a una rete neurale (specialmente in PyTorch), devi organizzarli in un formato matematico preciso chiamato Tensore:

- $N$ (Batch Size): È il numero di "esperienze" che la rete guarda contemporaneamente durante l'addestramento (es. 32 o 64 sequenze di immagini).
- $C$ (Channels): Qui sta il trucco. Se usiamo il Frame Stacking, i "canali" non sono più i colori (RGB), ma i frame temporali. Se stackiamo 4 frame in bianco e nero, $C$ sarà 4.
- $H$ (Height) & $W$ (Width): L'altezza e la larghezza in pixel (es. 84x84).

In sintesi: Per l'IA, lo stato è un "cubo" di informazioni dove ogni fetta è un momento diverso nel tempo.

### 1.2 Preprocessing: Trasformare i Pixel in Informazione

I pixel grezzi contengono troppo rumore. Ecco come li puliamo>

- Cropping (Ritaglio)
    - **Perchè?:** Riduce il numero di dati da processare e impedisce alla rete di "fissarsi" su elementi statici
- Grayscale (Scala di Grigi)
    - **Perchè?:** Riduce il numero di dati passando da 3 canali (RGB) ad 1 canale, La distinzione dei colori è computazionalmente costosa. Se il colore non è una variabile decisionale critica, lo eliminiamo
- Normalizzazione $[0, 1]$: I pixel originali sono uint8 (interi da 0 a 255). Li convertiamo in float32 e dividiamo per 255.
    - **Perchè:** Le funzioni di attivazione delle reti neurali (come ReLU o Sigmoid) lavorano meglio con input piccoli. Valori come "255" farebbero "esplodere" i gradienti, rendendo l'addestramento instabile o impossibile.

### 1.3 Temporalità: Il Frame Stacking ($k=4$)

Questo è il concetto più profondo. Una singola foto (frame statico) è un MDP (Markov Decision Process) incompleto, 

L'esempio del proiettile:

- Se vedi una foto di un proiettile a mezz'aria, non sai se sta venendo verso di te o se si sta allontanando. Non conosci la sua velocità.
- Se vedi 4 frame consecutivi, la rete neurale può confrontare la posizione del proiettile tra il frame 1 e il frame 4. La differenza di posizione nei pixel indica direzione e velocità.

Sovrapponendo i frame, trasformiamo una serie di immagini statiche in un flusso dinamico che contiene la fisica dell'ambiente.

### 1.4 Rappresentazione Visiva dello Stato

Immagina lo stato finale che inviamo alla rete:

- Prendi il frame attuale, taglia l'arma, fallo 84x84, trasformalo in grigio.
- Mettilo in una "coda" (buffer) insieme ai 3 frame precedenti.
- Unisci tutto in un unico blocco di dati.

**Risultato:** Un oggetto matematico che dice all'IA: "Ecco come appaiono le cose adesso e come si sono mosse negli ultimi frazioni di secondo".

Senza questo processo, la rete vedrebbe solo "rumore colorato". Con questo processo, la rete riceve segnali di movimento.

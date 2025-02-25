# Poetic Text Generator Using RNN 

This project demonstrates the generation of poetic text using a Recurrent Neural Network (RNN) built with LSTM (Long Short-Term Memory) layers in TensorFlow/Keras. The text generator is trained on a corpus of Shakespearean text to create new text that mimics the poetic style of Shakespeare.

## Table of Contents 📚
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Text Generation](#text-generation)
- [Helper Functions](#helper-functions)
- [How to Use](#how-to-use)
- [License](#license)

## Overview 

This project uses a text corpus (specifically, Shakespeare's works) and trains a model to generate new text that mimics the style and structure of Shakespearean poetry. The model is based on an RNN architecture with LSTM layers, which is a popular choice for sequence-based tasks like text generation.

The text generation model was trained on sequences of characters, where each character's presence at a given position in a sequence is encoded as a one-hot vector. After training, the model can predict the next character based on a given input sequence.

## Dependencies 📦

To run this project, you will need the following libraries:

- `tensorflow`
- `numpy`
- `random`

You can install these dependencies using pip:

```bash
pip install tensorflow numpy
```

## Data Preparation 📑

The model uses Shakespeare's text as a training dataset. The data is preprocessed as follows:
1. **Text Loading**: The Shakespeare corpus is downloaded and read into the program.
2. **Character Mapping**: Every unique character in the text is mapped to an index, and vice versa.
3. **Sequence Generation**: Sequences of characters are extracted from the text, each of length `SEQ_LENGTH`. The sequences are then used to predict the next character.

## Model Architecture 🧠

The model consists of:
- **LSTM Layer**: A layer with 128 units that processes the input sequences.
- **Dense Layer**: A fully connected layer that outputs the probability distribution of the next character.
- **Softmax Activation**: This activation function is used in the output layer to produce a probability distribution over the characters.

```plaintext
Model architecture:
1. LSTM (128 units)
2. Dense (Output size: number of characters)
3. Softmax activation
```

The model is trained using the **categorical cross-entropy** loss function and the **RMSprop** optimizer with a learning rate of 0.01.

## Training 🏋️‍♂️

The model is trained on the sequences for 4 epochs with a batch size of 256. After training, the model is saved as `Poetic_Text_Generator.h5`, which can be loaded for text generation later.

```plaintext
Training parameters:
- Batch size: 256
- Epochs: 4
- Optimizer: RMSprop (learning rate = 0.01)
- Loss function: Categorical cross-entropy
```

## Text Generation ✍️

After training, the model can generate new poetic text by predicting one character at a time based on an initial input sequence. The temperature parameter in the generation function allows you to control the randomness of the output. Lower temperature values make the model's predictions more deterministic, while higher temperatures introduce more randomness.

## Helper Functions 🛠️

- **`sample(preds, temperature=1.0)`**: This function takes the predicted character probabilities and adjusts them based on the temperature. It returns the index of the most likely next character.
  
- **`generate_text(length, temp)`**: This function generates a sequence of text with a specified length and temperature, starting from a random sequence in the text.

## How to Use 

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip`.
3. Run the script to train the model (if the model isn't already trained) or load the trained model and generate new text.
4. Play around with different temperature values to see how they affect the generated text!

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute or suggest improvements! 😊

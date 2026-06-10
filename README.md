# 🎵 AI Music Generation using LSTM

This project uses Artificial Intelligence and Deep Learning to generate new music from MIDI datasets. The model learns musical patterns from classical piano compositions and creates new melodies using an LSTM (Long Short-Term Memory) neural network.

## 📌 Project Overview

The system performs the following steps:

1. Collect MIDI music files (Classical Music Dataset)
2. Extract notes and chords using Music21
3. Preprocess musical sequences for training
4. Build a Deep Learning model using LSTM
5. Train the model on MIDI note sequences
6. Generate new music patterns
7. Save generated music as a MIDI file

---

## 🚀 Technologies Used

- Python 3.11
- TensorFlow / Keras
- Music21
- NumPy
- MIDI Files

---

## 📂 Project Structure

```text
CodeAlpha_MusicGenerationAI/
│
├── dataset/
│   ├── song1.mid
│   ├── song2.mid
│   └── ...
│
├── main.py
├── music_model.h5
├── generated_music.mid
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
https://github.com/saicharanrallabandi98-beep/Music-Generation-With-AI.git
```

### Install Dependencies

```bash
pip install tensorflow
pip install music21
pip install numpy
```

---

## 🎼 Dataset

The model is trained using classical MIDI files from composers such as:

- Beethoven
- Mozart
- Bach

All MIDI files are stored inside the `dataset` folder.

---

## 🧠 Model Architecture

The project uses an LSTM-based Recurrent Neural Network.

### Architecture

- LSTM (256 units)
- Dropout (0.3)
- LSTM (256 units)
- Dense (128 units)
- Dropout (0.3)
- Dense Output Layer (Softmax)

Total Parameters: ~837K

---

## 🏋️ Training

Run:

```bash
py -3.11 main.py
```

The model learns musical note patterns and saves the trained network as:

```text
music_model.h5
```

---

## 🎹 Music Generation

After training, the model generates new note sequences and creates:

```text
generated_music.mid
```

This file can be played using:

- VLC Media Player
- MIDI Player Applications
- Online MIDI Players

---

## 📊 Results

- Successfully extracted 6356 musical notes
- Generated 6256 training patterns
- Trained LSTM network for music generation
- Generated new AI-composed MIDI music

---

## 🎯 Learning Outcomes

Through this project:

- Learned MIDI data processing
- Applied sequence modeling using LSTM
- Built a Deep Learning music generator
- Generated original AI-composed music

---

## 👨‍💻 Author

**Ssai charan Rallabandi**

Project completed as part of AI/ML learning and practical implementation of Deep Learning for Music Generation.

---

## ⭐ Future Improvements

- Larger MIDI datasets
- Transformer-based music generation
- Multi-instrument support
- MIDI to MP3 conversion
- Web application deployment
- Real-time music generation

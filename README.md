
# TinyML Independent Study 
This project consists of a custom TinyML model which distinguishes voice by gender using mel-frequency cepstral coefficients (MFCCs). Training with MFCCs made the model more sound for recognition in human speech. I also took notes on a snow forecast model and my notes are also included. This project was developed for the Arduino Nano 33 BLE Sense Rev2 with headers. Below is an overview of the model I developed: 

## Short Summary
**TinyML Independent Study: a custom TinyML model which distinguishes voice by gender using mel-frequency cepstral coefficients (MFCCs)**

# Voice Gender Classification using LSTM

This project implements a binary audio classification system that predicts whether a voice recording belongs to a boy or a girl using MFCC features and a Bidirectional LSTM neural network. The pipeline includes audio preprocessing, feature extraction, sequence modeling, and evaluation on unseen audio samples.

The goal of this project is to explore sequence modeling on audio data using deep learning and to understand how recurrent neural networks can be applied to speech-related classification tasks.

---

## Project Overview

The system follows this pipeline:

1. Load `.wav` audio files  
2. Normalize audio signals  
3. Extract MFCC features  
4. Pad or truncate all samples to a fixed number of frames  
5. Feed MFCC sequences into a Bidirectional LSTM network  
6. Train a binary classifier  
7. Test the trained model on a new audio file  

Each audio clip is represented as a matrix of:

```
13 MFCC coefficients × 100 frames
```

After preprocessing, the data is reshaped to:

```
(number_of_samples, 100, 13)
```

which corresponds to:

```
(batch_size, time_steps, features)
```

---

## Model Architecture

The model is implemented using Keras and consists of:

- Bidirectional LSTM layer (4 units, return_sequences=True)  
- Bidirectional LSTM layer (32 units)  
- Dense output layer with sigmoid activation  

This architecture allows the network to process the MFCC sequence both forward and backward in time, which is useful since MFCC frames are not strictly causal in the same way as text or time-series signals.

The model is compiled with:

- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Metric: Accuracy  

---

## Dataset Structure

Training audio files are expected to be in a directory such as:

```
training/
├── boy_1.wav
├── boy_2.wav
├── girl_1.wav
├── girl_2.wav
├── ...
```

Labels are inferred from filenames:

- Files containing "boy" → label = 0  
- Files containing "girl" → label = 1  

---

## Dependencies

This project uses:

- Python 3  
- numpy  
- pandas  
- librosa  
- matplotlib  
- seaborn  
- scikit-learn  
- tensorflow / keras  
- torchaudio  

Install everything with:

```bash
pip install numpy pandas librosa matplotlib seaborn scikit-learn tensorflow keras torchaudio
```

---

## How to Run

1. Set the training directory path in the script:

```python
base_dir = r"path\to\training"
```

2. Run the script:

```bash
python train_lstm_audio.py
```

3. The script will:
- Load and preprocess all audio files  
- Extract MFCC features  
- Train the LSTM model  
- Print training and validation accuracy  

---

## How to Test on a New Audio File

At the bottom of the script, set:

```python
base_dir = r"path\to\test_audio_folder"
audio_file_path = os.path.join(base_dir, "your_test_file.wav")
```

Then run the script again. The output will be:

```python
predictions = model.predict(mfccs_test)
```

The result is a probability:

- Close to 0 → predicted "boy"  
- Close to 1 → predicted "girl"  

---

## Feature Extraction Details

- MFCCs are extracted using librosa  
- 13 coefficients per frame  
- Each sample is padded or truncated to 100 frames  
- All audio is normalized before feature extraction  

---

## Train/Test Split

The dataset is split using:

```python
train_test_split(test_size=0.2, stratify=y)
```

This ensures both classes remain balanced in both sets.

---

## Notes and Limitations

- The dataset is small, so overfitting is possible  
- The model is intentionally kept small  
- This project is meant as a learning experiment in audio ML and LSTM sequence modeling, not a production system  
- Performance depends heavily on dataset size and recording quality  

---

## Possible Improvements

- Use a larger dataset  
- Add data augmentation  
- Try CNN + LSTM hybrid architecture  
- Tune MFCC parameters and sequence length  
- Add k-fold cross validation  
- Save and reload trained models  

---

## Example Output

After training, the model prints predictions such as:

```
[[0.87]]
```

Which would correspond to a high confidence prediction for the "girl" class.





## Textbooks Used
TinyML Cookbook - Second Edition By Gian Marco Iodice
Tiny Machine Learning Quickstart: Machine Learning for Arduino Microcontrollers By Simone Salerno
TinyML By Pete Warden, Daniel Situnayake



# Speech Enhancement Pipeline (Why Classical Methods Aren't Enough)

## Overview

This repository implements an end-to-end speech enhancement pipeline that utilises both classical signal processing techniques and modern deep learning models to explore the limitations of each method and their efficiency in enhancing speech signals.
The framework uses the DARPA TIMIT dataset for training and evaluation, and augments the audio files with various noise types to simulate realistic background interference. The augmented audio files and the third-party models used in this implementation
aren't included in this repository due to licensing restrictions, but instructions for obtaining them and relevant links are provided.

The implemented algorithms include:
-Wiener Filtering
-Spectral Subtraction
-CNN-based U-Net Enhancemnet
-DeepFilterNet
-FullSubNet

The pipeline is designed to be modular and has several components that support dataset preparation, noise generation, feature extraction, CNN training, third party integration, batch enhancement (according to the model), spectrogram visualization, and model evaluation.
The evaluation metrics include Signal-to-Noise Ratio (SNR), Scale-Invariant Signal-to-Distortion Ratio (SI-SDR/SI-SNR), and Short-Time Objective Intelligibility (STOI).

After evaluation and comparison of the results, it was found that the neural network-based models significantly outperformed the classical methods in terms of both objective metrics and subjective listening tests. The proposed and implemented U-Net CNN outperformed FullSubNet (and DeepFilterNet), while the classical enhancement techniques had comparable performance. The classical enhancement techniques, however, have a far lower computational cost and are still relevant methods of investigation

<figure align="center">
  <img src="./reports/plots/mean_SI_SDR" width="500">
  <figcaption>Figure 1: The average SI-SDR values recorded by each enhancement method. FullSubNet recorded the best SI-SDR value of 15.9389 followed by the U-Net CNN with a value of 15.8377.</figcaption>
</figure>


<figure align="center">
  <img src="./reports/plots/mean_SNR" width="500">
  <figcaption>Figure 2: The average SNR values recorded by each enhancement method. FullSubNet recorded the best SNR value of 16.864 followed by the U-Net CNN with a value of 16.7698.</figcaption>
</figure>


<figure align="center">
  <img src="./reports/plots/mean_STOI" width="500">
  <figcaption>Figure 3: The average STOI values recorded by each enhancement method. The implemented U-Net CNN recorded the best STOI value of 0.9353 followed by FullSubNet with a value of 0.9336.</figcaption>
</figure>

The comparative reports and plots can be found in /reports under the project root. The implemented CNN also automatically generates spectrograms for the predicted mask, noisy audio file, and the enhanced audio file when run. These spectrograms will be saved to the 
outputs/spectrograms/cnn folder when batch_enhancement.py is run with the following arguments: python -m scripts.batch_enhancement --model cnn --noise white --snr 10dB --split test

The other enhancement models can be run with similar arguments, where the only difference would be the model (fullsubnet, deepfilternet, spectral, or wiener), the noise type (white or babble), the snr type (which is a string), and the split (test,train,or val). The same can 
be followed for the evaluate_model.py file which evaluates the several models against the clean audio files for ranking. The FullSubNet model was run locally by downloading both the checkpoints and the actual model from their official releases. The officially released model checkpoint "fullsubnet_best_model_58epochs.tar" was used. DeepFilterNet was used after a simple installation
in the terminal. A local Python virtual environment of Python 3.11.9 was used. All the resources used and information on how to save them is mentioned below.

The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus can be downloaded from: <https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech>. The extracted folder can be saved as TIMIT under the datasets folder. Running the dataset preparation script immediately after should create the data folder under the project root
with seperate folders for clean, noisy, and enhanced speech. The directory structure from the original TIMIT folder should be preserved. If you'd like to test the model against a different dataset, you can add audio files to data/noisy/white or data/noisy/babble for enhancement.

The FullSubNet+ model that was initially downloaded to a third_party folder under the project root can be downloaded from: <https://github.com/RookieJunChen/FullSubNet-plus> and the official checkpoint can be found at: <https://github.com/Audio-WestlakeU/FullSubNet/releases>. The checkpoint for this model is saved under the pretrained_models/fullsubnet folder. The DeepFilterNet model can simply be installed by running the pip install deepfilternet command in the terminal. 
The pretrained model for DeepFilterNet is automatically downloaded when the model is run for the first time and you can simply use the deepfilternet argument in batch_enhancement.py. 

The checkpoints for the implemented U-Net CNN model are saved under the pretrained_models/cnn folder. The model was trained for 100 epochs with a batch size of 16 and a learning rate of 0.001. Due to time constraints, the model was only trained for 21 epochs for white noise and 16 epochs for babble noise.
The training script can be run by executing the following command in the terminal: python -m scripts.train_cnn --epochs 100 --batch_size 16 --learning_rate 0.001 and both the best and last checkpoints will be saved automatically.

The research paper that was used as inspiration and as a reference for the structure of the model can be found at: <https://zqwang7.github.io/publications/icassp2017_4.pdf>. The experimental methodology of the implemented CNN was inspired by the Adaptive Speech Spectrogram Approximation (ASSA) framework outlined in X. Zhang et al's paper "A Speech Enhancement Algorithm By Iterating Single and Multi-Microphone Processing and Its Application to Robust ASR". ASSA formulates speech enhancement as a supervised learning problem in which a neural network estimates an Ideal Ratio Mask (IRM) from noisy speech spectrograms. Rather than reproducing the original ASSA architecture exactly, I chose to adopt the overall experimental philosophy while replacing the enhancement algorithm with modern neural architectures including DeepFilterNet, FullSubNet, and a custom U-Net CNN developed specifically for this project.
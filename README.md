# Inference

To run inference
1. Go to inference folder
2. Run inference.ipynb file in sequence
3. It will download the model weights from Google drive 

# Training code is available in the `training` folder

Contains experiments including:
1. Experimenting with different novelty rewards (a lot of them is too noisy and doesn't converge) 
2. Experiments with different models such as LSTM and different encodings
3. Getting information from depth sensor (RGB-D) image
4. Classical object estimation based on stereo view
5. FAISS similarity that works out of the box (Omitted in the report as it works without any fine tuning)

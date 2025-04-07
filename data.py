from datasets import load_dataset
import os
import numpy as np
import torch
import torchaudio.transforms as transforms
from tqdm import tqdm
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------------
# ---- Create "data" Directory for Audio and Labels ----

# Create "data" directory to hold "audio" and "labels"
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

# Create "audio" and "labels" subfolders inside "data"
audio_dir = os.path.join(data_dir, "audio")
labels_dir = os.path.join(data_dir, "labels")
# Create the "audio" and "labels" directories if they don't exist
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# ----------------------------------------------------------------------------
# ---- Functions for Loading Split and Preparing Spectrograms and Labels ----


def load_split(split):
    # https://huggingface.co/datasets/google/speech_commands
    ds = load_dataset("google/speech_commands", 
                    "v0.02",
                    split=split,
                    trust_remote_code=True)
    ds = ds.shuffle(seed=1337)

    # Use a total of 10 shards each for spectrograms and their labels
    shard_size = int(np.ceil(len(ds) / 10))

    print(f"Number of examples for each class in {split} split: {list(Counter(ds["label"]).values())}")

    return ds, shard_size


def tokenize(sample):
    # Get the audio sample and sampling rate
    audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
    sampling_rate = sample["audio"]["sampling_rate"]

    # Convert to PyTorch tensor
    waveform = torch.tensor(
        audio_array, device=device).unsqueeze(0) # (1, Time)
    
    # Pad/truncate the audio sample to 1 second at 16 kHz
    target_length = sampling_rate
    waveform_length = waveform.shape[-1]

    if waveform_length < target_length:
        # Pad the waveform with zeros (at the end)
        padding = target_length - waveform_length
        waveform = torch.cat(
            [waveform, torch.zeros((1, padding), device=device)], dim=-1)
        
    elif waveform_length > target_length:
        # Skip the audio sample and its label if longer than 1 sec
        return None, None

    # Compute Mel Spectrogram
    sample_rate = 16000
    n_fft = int(25 / 1000 * sample_rate)  
    hop_length = int(10 / 1000 * sample_rate)
    n_mels = 80  
    mel_spectrogram_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    ).to(device)

    mel_spectrogram = mel_spectrogram_transform(waveform) # (1, 80, Time)
    
    # Convert to log scale
    log_mel_spectrogram = torch.log10(mel_spectrogram + 1e-6)

    label = sample["label"]

    return log_mel_spectrogram, label

# ----------------------------------------------------------------------------
# ---- Create Spectrogram and Label Shards and Save to File ----


def main():
    for split in ["train", "validation", "test"]:
        ds, shard_size = load_split(split)
        shard_index = 0
        example_count = 0
        progress_bar = None
        spectrograms = []
        labels = []
        position = 0
        total = len(ds)

        for sample in ds:
            position += 1
            
            log_mel_spectrogram, label = tokenize(sample)
            
            if log_mel_spectrogram == None:
                continue

            # Create a progress bar for this pair of shards
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size,
                                    unit=" Examples",
                                    desc=f"Shard {shard_index}")
                
            spectrograms.append(log_mel_spectrogram)
            labels.append(label)
            example_count += 1

            # update progress bar
            progress_bar.update(1)

            # Save the pair of shards if they have been filled
            if example_count == shard_size or position == total:

                # Save spectrograms shard to "audio" directory
                audio_tensor = torch.cat(spectrograms)
                audio_path = os.path.join(
                    audio_dir, f"{split}_audio_{shard_index:02d}.pt")
                torch.save(audio_tensor.cpu(), audio_path)

                # Save labels shard to "labels" directory
                labels_tensor = torch.tensor(labels)
                labels_path = os.path.join(
                    labels_dir, f"{split}_labels_{shard_index:02d}.pt")
                torch.save(labels_tensor.cpu(), labels_path)

                shard_index += 1
                example_count = 0
                progress_bar = None
                spectrograms = []
                labels = []

if __name__ == '__main__':
    main()
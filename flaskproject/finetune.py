import os
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
from torch.nn.utils.rnn import pad_sequence

# Dataset class to load audio files and their transcriptions
class AudioDataset(Dataset):
    def __init__(self, audio_dir, processor, tokenizer):
        self.audio_dir = audio_dir
        self.processor = processor
        self.audio_files = self.load_audio_files(audio_dir)
        print(self.audio_files)
        self.transcriptions = self.load_transcriptions(audio_dir)        
        self.tokenizer = tokenizer

    # def load_transcriptions(self, transcription_file):
    #     with open(transcription_file, 'r') as f:
    #         content = f.read()  # Read the entire file
    #     return content

    def load_transcriptions(self, audio_dir):
        transcriptions = {}
        for audio_file in self.audio_files:
            # Assuming audio files are in .mp3 format and text files are .txt
            base_name = os.path.splitext(audio_file)[0]
            transcription_file = os.path.join(audio_dir, f"{base_name}.txt")
            if os.path.exists(transcription_file):
                with open(transcription_file, 'r') as f:
                    transcriptions[audio_file] = f.read()  # Read and store transcription
        return transcriptions

    def load_audio_files(self, audio_dir):
        return [f for f in os.listdir(audio_dir) if f.endswith('.mp3') or f.endswith('.wav')]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]  # Get the audio file name
        transcription_file = self.transcriptions[audio_file] # Get the specific transcription for the current index
        # print(audio_file, transcription)
        # Tokenize the transcription
        transcription = self.tokenizer(transcription_file, return_tensors="pt").input_ids # Shape (N,)
        
        audio_path = os.path.join(self.audio_dir, audio_file)
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.dim() == 2:  # Stereo audio
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        # Process the audio waveform
        inputs = self.processor(waveform.squeeze(0), return_tensors="pt", truncation=True, padding="longest", return_timestamps=True, return_attention_mask=True, sampling_rate=16_000)
        
        # Ensure that inputs['input_ids'] is accessed correctly
        input_ids = inputs.input_features.squeeze(0)  # Shape (M,) where M is the number of input tokens
        print(f"Input IDs shape: {input_ids.shape}, Transcription shape: {transcription.shape}")
        
        return input_ids, transcription.squeeze(0)  # Return both as tensors

# Function to fine-tune the Whisper model
def collate_fn(batch):
    inputs, transcriptions = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True)  # Pad input IDs
    transcriptions = pad_sequence(transcriptions, batch_first=True)  # Pad transcriptions
    return inputs, transcriptions


def fine_tune_whisper(audio_dir, model, processor, tokenizer, num_epochs=3, batch_size=4):
    dataset = AudioDataset(audio_dir, processor, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, transcriptions = batch
            # labels = processor(transcriptions, return_tensors="pt", padding=True).input_ids
            labels = transcriptions

            optimizer.zero_grad()
            outputs = model(input_features=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
    model.save_pretrained("../new/model")
    processor.save_pretrained("../new/model")
    return 'success'

# Main execution
def start_fine_tune(data_directory):
    model_name = "openai/whisper-base"
    processor = WhisperProcessor.from_pretrained(model_name, language="german", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to('cuda')
    tokenizer = WhisperTokenizer.from_pretrained(model_name)

    # Fine-tune the model
    result = fine_tune_whisper(data_directory, model, processor, tokenizer)
    return result

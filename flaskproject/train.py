import os
import torchaudio
import torch
from transformers import WhisperForConditionalGeneration, AutoProcessor

def transcribe_audio(file_path):
    # Load the model and processor
    model = WhisperForConditionalGeneration.from_pretrained("../new/model")
    processor = AutoProcessor.from_pretrained("../new/model")

    # Load audio
    audio_input, sample_rate = torchaudio.load(file_path)

    # Resample if necessary
    if sample_rate != 16_000:
        audio_input = torchaudio.functional.resample(audio_input, sample_rate, 16_000)

    # Convert to mono if stereo
    if audio_input.shape[0] == 2:
        audio_input = audio_input.mean(dim=0, keepdim=True)

    # Process the audio input for the model
    inputs = processor(audio_input.squeeze(0), return_tensors="pt", truncation=False, padding="longest", return_timestamps=True, return_attention_mask=True, sampling_rate=16_000)

    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features, return_timestamps=True, return_token_timestamps=True)

    # Decode the transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription

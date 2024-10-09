import subprocess
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
import os
import torchaudio
import torch.nn.functional as F

model = WhisperForConditionalGeneration.from_pretrained("../new/model")
processor = WhisperProcessor.from_pretrained("../new/model")
tokenizer = WhisperTokenizer.from_pretrained("../new/model")
audio_file = "1.wav"
cut_duration = 30
overlap_duration = 5

def get_audio_duration(audio_file):
    command = ["ffmpeg", "-i", audio_file]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    duration_line = [line for line in result.stderr.splitlines() if "Duration" in line]
    if duration_line:
        duration_str = duration_line[0].split(",")[0].split("Duration: ")[1]
        h, m, s = map(float, duration_str.split(":"))
        return int(h * 3600 + m * 60 + s)
    return 0

total_duration = get_audio_duration(audio_file)
print(f"Total duration: {total_duration} seconds")
start_times = []
i = 0
while True:
    start_time = (cut_duration - overlap_duration) *i
    if start_time >= total_duration:
        break
    start_times.append(start_time)
    i += 1

def cut_audio_segments(audio_file, start_times, duration):
    for start in start_times:
        output_file = f"output/segment_{start}_{start + duration}.wav"
        if (start + duration > total_duration):
            output_file = f"output/segment_{start}_{total_duration}.wav"
        command = [
            "ffmpeg", "-i", audio_file, "-ss", str(start), "-t", str(duration), "-c", "copy", output_file
        ]
        subprocess.run(command)
def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"
cut_audio_segments(audio_file, start_times, cut_duration)

def transcribe_audio_with_timestamps(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True) 

    inputs = processor(waveform.squeeze(0), return_tensors="pt", truncation=True, padding="longest", return_timestamps=True, return_attention_mask=True, sampling_rate=16_000)
    audio_input = inputs.input_features
    if audio_input.shape[-1] < 3000:
        padding = 3000 - audio_input.shape[-1]
        audio_input = F.pad(audio_input, (0, padding), "constant", 0)

    if audio_input.ndim == 2:
        audio_input = audio_input.unsqueeze(0) 
    with torch.no_grad():
        predicted_ids = model.generate(audio_input, return_dict_in_generate=True, return_token_timestamps=True, language="german")
    transcription = processor.batch_decode(predicted_ids.sequences, skip_special_tokens=True)[0]
    token_timestamps = predicted_ids.token_timestamps.squeeze(0)
    tokens = processor.tokenizer.tokenize(transcription)
    words = transcription.split()
    word_timestamps = []

    if token_timestamps.ndim == 2:
        temp = ''
        start_time = None
        end_time = None
        for token, (start, end) in zip(tokens, token_timestamps):
            if ('Ġ' in token):
                if (temp):
                    word_timestamps.append((temp, start_time, end_time))
                temp = token.replace('Ġ', '')
                start_time = start
                end_time = end
            else:
                temp += token
                end_time = end
        word_timestamps.append((temp, start_time, end_time))
    else:
        temp = ''
        start_time = None
        end_time = None
        for token, timestamp in zip(tokens, token_timestamps):
            if ('Ġ' in token):
                if (temp):
                    word_timestamps.append((temp, start_time, end_time))
                temp = token.replace('Ġ', '')
                start_time = timestamp.item()
                end_time = timestamp.item()
            else:
                temp += token
                end_time = timestamp.item()
        word_timestamps.append((temp, start_time, end_time))
    
    for i in range(len(word_timestamps)):
        word_timestamp = word_timestamps[i]
        current_word = word_timestamp[0]  # Get the word from words_timestamps
        if i < len(words):  # Ensure we do not exceed the bounds of the words list
            expected_word = words[i]  # Get the corresponding word from words
            if current_word != expected_word:
                # Replace the first element of the tuple in words_timestamps
                word_timestamps[i] = (expected_word, word_timestamp[1], word_timestamp[2])
    return transcription, word_timestamps

full_transcription = []
transcription_only = []

for start in start_times:
    end = start + cut_duration 
    if (end > total_duration):
        end = total_duration
    segment_file = f"output/segment_{start}_{end}.wav"
    if os.path.exists(segment_file):
        transcription, word_timestamps = transcribe_audio_with_timestamps(segment_file)
        if (full_transcription):
            last_entry = full_transcription[-1].split('] ')[1]
            index = 0
            for i in range(len(word_timestamps)):
                word_timestamp = word_timestamps[i]
                # print(f"{last_entry[:2]}, {word_timestamp[0][:2]}")
                if (last_entry[:2] == word_timestamp[0][:2]):
                    index = i
                    break
            full_transcription = full_transcription[:-(index + 1)]
            if (transcription_only):
                last_transcription = transcription_only[-1]
                words = last_transcription.split()
                words = words[:-(index + 1)]
                words = ' '.join(words) 
                transcription_only = transcription_only[:-1]
                transcription_only.append(words)
        for word, start_time, end_time in word_timestamps:
            full_transcription.append(f"[{format_time(start + start_time)} - {format_time(start + end_time)}] {word}")
        transcription_only.append(transcription)

with open("full_transcription_with_timestamps.txt", "w", encoding="utf-8") as f:
    for line in full_transcription:
        f.write(line + "\n")
with open("full_transcription_only.txt", "w", encoding="utf-8") as f:
    for line in transcription_only:
        f.write(line + "\n")

print("Transcription completed and saved to full_transcription_with_timestamps.txt.")

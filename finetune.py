# Import necessary libraries
from datasets import Dataset, Audio
import pandas as pd
import gc
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, AutoProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import torch
import torchaudio
from dataclasses import dataclass
from typing import Any, Dict, List, Union

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
# -------------------- Data Loading and Preprocessing --------------------

# Load the training and testing data
train_df = pd.read_csv("data.csv")
test_df = pd.read_csv("test.csv")

# Rename the columns to "audio" and "sentence"
train_df.columns = ["audio", "sentence"]
test_df.columns = ["audio", "sentence"]

# Convert the pandas dataframes to Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Convert the sample rate of every audio file
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

# -------------------- Feature Extraction and Tokenization --------------------

# Import feature extractor and tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="English", task="transcribe")


# Prepare the dataset by extracting features and encoding labels
def prepare_dataset(examples):
    # Compute log-Mel input features from input audio array
    audio = examples["audio"]
    examples["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    del examples["audio"]

    sentences = examples["sentence"]
    # Encode target text to label ids
    examples["labels"] = tokenizer(sentences).input_ids
    del examples["sentence"]
    return examples


# Map the prepare_dataset function to the datasets
train_dataset = train_dataset.map(prepare_dataset, num_proc=1)
test_dataset = test_dataset.map(prepare_dataset, num_proc=1)


# -------------------- Data Collation --------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different lengths and padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Cut bos token if appended in previous tokenization step
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# Initiate the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# -------------------- Metrics Computation --------------------

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute Word Error Rate (WER)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# -------------------- Model Initialization and Training Configuration --------------------

# Load the model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False
#device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#print(model)

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-base-ge",  # change to a repo name of your choice
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-7,
    warmup_steps=6,
    max_steps=40,
    gradient_checkpointing=True,
    fp16=False,
    eval_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=5,
    eval_steps=5,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)
#print(training_args)
# -------------------- Trainer Initialization and Training --------------------

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
print(trainer)

# Start the model training
trainer.train()

model.save_pretrained("new/model")
processor.save_pretrained("new/model")

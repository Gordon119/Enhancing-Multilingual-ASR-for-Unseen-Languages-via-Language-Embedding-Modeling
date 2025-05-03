import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import WhisperForConditionalGeneration
from datasets import load_dataset
import matplotlib.pyplot as plt
from iso639 import Lang
from typing import Optional, Union
from transformers.modeling_outputs import BaseModelOutput
from transformers.generation.configuration_utils import GenerationConfig
import numpy as np
from transformers import WhisperProcessor
import nlp2
import torchaudio
from tqdm import tqdm
from pytorch_metric_learning import samplers


LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

LANGUAGES_2_ID = {v: k for k, v in LANGUAGES.items()}

def prepare_dataset_whisper(batch, feature_extractor, audio_feature_key):
    path = batch["path"]
    speech, sampling_rate = torchaudio.load(path)
    if sampling_rate != "16_000" or sampling_rate != "16000":
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        batch[audio_feature_key] = resampler.forward(speech.squeeze(0)).numpy()
    else:
        batch[audio_feature_key] = speech.squeeze(0).numpy()
    # compute log-Mel input features from input audio array
    batch[audio_feature_key] = feature_extractor(batch[audio_feature_key], sampling_rate=16000).input_features[0]
    batch["lengths"] = len(batch[audio_feature_key])
    # # encode target text to label ids
    # batch["labels"] = tokenizer(batch["sentence"]).input_ids
    if "sentence" in batch:
        batch["labels"] = batch["sentence"]
    else:
        batch["labels"] = batch["text"]
    return batch

def encode_dataset(batch, processor, phonemize=False, backend=None, separator=None):
    if not isinstance(batch["labels"], list):
        if phonemize:
            with processor.as_target_processor():
                if phonemize == "g2p":
                    batch["labels"] = processor(backend.encode(batch["labels"])).input_ids
                else:
                    batch["labels"] = processor(backend.phonemize([batch["labels"]], separator=separator)[0]).input_ids
        else:
            try:
                with processor.as_target_processor():
                    line = bytes(batch["labels"], "utf-8").decode("utf-8", "ignore")
                    batch["labels"] = processor(line).input_ids
            except Exception as e:
                line = bytes(batch["labels"], "utf-8").decode("utf-8", "ignore")
                batch["labels"] = processor.tokenizer(line).input_ids
    if len(batch["labels"]) > 448:
        batch["labels"] = batch["labels"][:448]
    return batch

class Whisper_Modified(WhisperForConditionalGeneration):
    def detect_language_custom(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Union[torch.FloatTensor, BaseModelOutput]] = None,
        generation_config: Optional[GenerationConfig] = None,
        num_segment_frames: int = 3000,
        mask: str = None
    ) -> torch.Tensor:
        if input_features is None and encoder_outputs is None:
            raise ValueError("You have to specify either `input_features` or `encoder_outputs`")
        elif input_features is not None and encoder_outputs is not None:
            raise ValueError("Make sure to specificy only one of `input_features` or `encoder_outputs` - not both!")
        elif input_features is not None:
            inputs = {"input_features": input_features[:, :, :num_segment_frames]}
            batch_size = input_features.shape[0]
        elif encoder_outputs is not None:
            inputs = {"encoder_outputs": encoder_outputs}
            batch_size = (
                encoder_outputs[0].shape[0] if isinstance(encoder_outputs, BaseModelOutput) else encoder_outputs[0]
            )

        generation_config = generation_config or self.generation_config
        decoder_input_ids = (
            torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
            * generation_config.decoder_start_token_id
        )

        with torch.no_grad():
            logits = self(**inputs, decoder_input_ids=decoder_input_ids).logits[:, -1]

        non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
        lang_id = list(generation_config.lang_to_id.values())
        mask_id = generation_config.lang_to_id[mask]
        non_lang_mask[lang_id] = False
        non_lang_mask[mask_id] = True
        logits[:, non_lang_mask] = -np.inf
        dist = logits.softmax(-1)[:, lang_id]
        return dist

def get_embedding(mask):
    input_arg = {}
    size = "large-v2"
    input_arg["tokenize_config"] = f"openai/whisper-{size}"
    input_arg["model_config"] = f"openai/whisper-{size}"
    input_arg["group_by_length"] = True
    input_arg["custom_set_train"] = f"data/ml_superb/sixth_edition/seen_languages/{mask}/train.csv"

    lang_name = Lang(mask).name.lower()
    model = Whisper_Modified.from_pretrained(input_arg["model_config"])

    for key, value in LANGUAGES.items():
        if value == lang_name:
            mask = f"<|{key}|>"
    import os
    processor = WhisperProcessor.from_pretrained(
        input_arg["model_config"], task="transcribe", language=lang_name
    )
    audio_feature_key = "input_ids"
    # load from base model
    
    model = model.to("cuda")
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    dataset = load_dataset(
        "csv",
        data_files=input_arg["custom_set_train"],
        cache_dir="~/.cache"
    )
    dataset = dataset.filter(lambda e: nlp2.is_file_exist(e["path"]))
    data_train = dataset["train"]
    data_train = data_train.map(
        prepare_dataset_whisper,
        num_proc=1,
        fn_kwargs={"feature_extractor": processor.feature_extractor, "audio_feature_key": audio_feature_key},
    )
    data_train = data_train.map(encode_dataset, fn_kwargs={"processor": processor})
    weight = list()
    for batch in tqdm(data_train):
        lang_distribution = model.detect_language_custom(torch.Tensor(batch["input_ids"]).unsqueeze(0).to("cuda"), mask=mask)
        weight.append(lang_distribution)
    return weight

def prepare_data():
    labels2class = dict()
    labels2embed = list()

    with torch.no_grad():
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
        embedding = model.get_decoder().get_input_embeddings()
        lang_id = torch.tensor(list(model.generation_config.lang_to_id.values()))
        lang_embedding = embedding(lang_id).to("cuda")
        train_x = list()
        train_y = list()
        import os
        lang_names = os.listdir("data/ml_superb/sixth_edition/seen_languages")

        cnt = 0
        for i in lang_names:
            i = LANGUAGES_2_ID[Lang(i).name.lower()]
            name = f"<|{i}|>"
            labels2class[i] = cnt
            labels2embed.append(embedding(torch.tensor(model.generation_config.lang_to_id[name])).detach())
            cnt += 1

        for idx, name in enumerate(lang_names):
            weights = get_embedding(name)
            name_ori = LANGUAGES_2_ID[Lang(name).name.lower()]
            target = labels2class[name_ori]
            for weight in weights:
                input = torch.matmul(weight, lang_embedding).detach()
                train_x.append(input)
                train_y.append(target)
    return torch.stack(train_x, dim=0), train_y, labels2class, labels2embed

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

from torch import Tensor
from typing import Tuple
class MLP_Dataset(Dataset[Tuple[Tensor, ...]]):
    def __init__(self, features, labels) -> None:
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return tuple((self.features[index], self.labels[index]))

    def __len__(self):
        return self.features.size(0)

import numpy as np
import torch

# X_train, y_train, labels2class, labels2embed = prepare_data()
X_train, y_train, labels2class, labels2embed = torch.load("weights_mlp_predictor_batch/all.pt")
# torch.save([X_train, y_train, labels2class, labels2embed], "weights_mlp_predictor_batch/all.pt")
y_train = torch.tensor(y_train)
X_train = X_train.squeeze(1).cpu()
train_dataset = MLP_Dataset(X_train, y_train)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])

torch.save(train_dataset, "weights_mlp_predictor_batch/train_dataset.pt")

class_sample_count = torch.bincount(y_train)
weights = 1 / torch.Tensor(class_sample_count)
# sample_weights = [weights[i] for i in torch.index_select(y_train, 0, train_indices)]
input_size = X_train.shape[-1]
output_size = 1280
epochs = 100
patience = 3
best_search_loss = float("inf")
best_config = {}

train_losses = []
val_losses = []

for hidden_size in [32, 64, 128, 256, 512, 1024, 1280]:
    for batch_size in [32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32]:
        for learning_rate in [0.00001, 0.00005, 0.00007, 0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007, 0.008, 0.009, 0.01, 0.03]:
            # sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, batch_size,replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            # Initialize the model, loss function, and optimizer
            model = MLP(input_size, hidden_size, output_size).to("cuda")
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

            from tqdm import tqdm
            # Training loop
            best_loss = float("inf")
            cnt = 0
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for inputs, targets in tqdm(train_loader):
                    optimizer.zero_grad()
                    inputs = inputs.to("cuda")
                    outputs = model(inputs)
                    targets = torch.stack([labels2embed[label] for label in targets])
                    targets = targets.to("cuda")
                    loss = criterion(outputs, targets)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                train_loss /= len(train_loader)
                train_losses.append(train_loss)

                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}")

                # Evaluation loop
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to("cuda")
                        outputs = model(inputs)
                        targets = torch.stack([labels2embed[label] for label in targets])
                        targets = targets.to("cuda")
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)

                if val_loss < best_loss:
                    cnt = 0
                    best_loss = val_loss
                    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")
                else:
                    cnt += 1
                    if cnt >= patience:
                        print(f"Early Stopping: Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")
                        break
            print("Complete.")
            if best_loss < best_search_loss:
                best_config = {
                    "lr": learning_rate,
                    "batch_size": batch_size,
                    "hidden_size": hidden_size
                }
                best_search_loss = best_loss
                torch.save(model, f"weights_mlp_predictor_batch/model.pt")

# Plot the learning curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title(f'Learning Curve (hidden_size={hidden_size}, batch_size={batch_size}, lr={learning_rate})')
plt.legend()
plt.savefig("weights_mlp_predictor_batch/learning_curve.png")

print("*******Search Results*******")
print(f"config: {best_config}")
print(f"loss: {best_search_loss}")
import json
json.dump({
    "config": best_config,
    "loss": best_search_loss
    }, open("weights_mlp_predictor_batch/model_info.json", "w"))

print("*******Verify Results*******")
pdist = nn.PairwiseDistance(p=2)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
model.eval()
with torch.no_grad():
    hit = 0
    for inputs, targets in train_loader:
        distances = list()
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        outputs = model(inputs).to("cuda")
        target_distance = pdist(outputs, targets)
        for value in labels2embed:
            value = value.to("cuda")
            distances.append(pdist(outputs, value))
        flag = 1
        for value in distances:
            if target_distance == value:
                continue
            if target_distance > value:
                flag = 0
                break
        hit += flag
    print(f"train_hit_rate:{hit/len(train_loader)}")
    hit_val = 0
    for inputs, targets in val_loader:
        distances = list()
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        outputs = model(inputs).to("cuda")
        target_distance = pdist(outputs, targets)
        for value in labels2embed:
            value = value.to("cuda")
            distances.append(pdist(outputs, value))
        flag = 1
        for value in distances:
            if target_distance == value:
                continue
            if target_distance > value:
                flag = 0
                break
        hit_val += flag
    print(f"val_hit_rate:{hit_val/len(val_loader)}")
import json
json.dump({"train_hit_rate":hit/len(train_loader), "val_hit_rate":hit_val/len(val_loader)}, open("weights_mlp_predictor_batch/match_rate.json", "w"))

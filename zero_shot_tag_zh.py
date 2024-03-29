# Run without int8, with fp16
from huggingface_hub import notebook_login

notebook_login()
import wandb
import copy
wandb.init(mode="disabled")
import sys

import nlp2
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperModel, WhisperConfig
from transformers import WhisperProcessor
from datasets import load_from_disk
from transformers.activations import ACT2FN
from module.args import parse_args
from module.data_processing import DataCollatorSpeechSeq2SeqWithPadding
from module.metric import cer_cal, wer_cal

from datetime import datetime


from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
import torch
import torchaudio
from typing import Callable, Iterator, List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.whisper.modeling_whisper import shift_tokens_right, WhisperDecoder, WhisperEncoder, WhisperPositionalEmbedding, WhisperDecoderLayer
from torch.nn import CrossEntropyLoss
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa

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
    # if batch["path"].split("/")[4] == "TD":
    #     batch["labels"][1] = 50260
    # else:
    #     batch["labels"][1] = 51865
    return batch

def experiment(input_arg, model, processor, data_collator, repo_name, data_train, data_test, time, output_dir):
    ###################
    #     Evaluate    #
    ###################
    eval_dataloader = DataLoader(data_test, batch_size=int(input_arg["batch"]), collate_fn=data_collator)

    model.eval()
    model = model.to("cuda")
    label_list = []
    pred_list = []
    pred_results = []
    
    for step, batch in enumerate(tqdm(eval_dataloader)): 
        with torch.no_grad():
            generated_tokens = (
                 model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )

            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            generated_tokens = torch.from_numpy(generated_tokens)
            pred_str = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)


            label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=False)

            pred_result = [[l, p, cer_cal([l], [p])] for l, p in zip(label_str, pred_str)]
            pred_results += pred_result

            pred_list += pred_str
            label_list += label_str
            pred_str = (" ").join(pred_str)

            if step == 0:
                print(pred_result)

        del generated_tokens, labels, batch
        gc.collect()
    nlp2.write_csv(pred_results, f'{output_dir}/pred.csv')
    cer = cer_cal(label_list, pred_list)
    wer = wer_cal(label_list, pred_list)
    print("********* Evaluation Result *********")
    print(f"cer: {cer}, wer: {wer}")
    # print(f"{lang_record}")
    print("*************************************")
    return model


def main(arg=None):
    from transformers import set_seed
    set_seed(42)
    input_arg, other_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    ############
    #  Config  #
    ############
    size = input_arg["size"]
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # input_arg["custom_set_train"] = "data/TD/train.csv"
    # input_arg["custom_set_test"] = "data/TD/test.csv"
    input_arg["tokenize_config"] = f"openai/whisper-{size}"
    input_arg["model_config"] = f"openai/whisper-{size}"
    input_arg["group_by_length"] = True
    input_arg["cache_dir"] = "/home/gordon1109/.cache"
    input_arg["epoch"] = 1
    dropout = input_arg.get("dropout", 0.0)

    repo_name = f"data/{input_arg['custom_set_train'].split('/')[1]}"

    ############
    #  Model   #
    ############

    processor = WhisperProcessor.from_pretrained(
        input_arg["model_config"], task="transcribe", dropout=dropout
        , language="Chinese"
    )
    # special_tokens_dict = {'additional_special_tokens': ['<|tat|>', '<|td|>'] + processor.tokenizer.all_special_tokens}
    # special_tokens_dict = {'additional_special_tokens': ['<|tat|>'] + processor.tokenizer.all_special_tokens}
    # num_added_toks = processor.tokenizer.add_special_tokens(special_tokens_dict)
    processor.save_pretrained(repo_name)
    audio_feature_key = "input_ids"
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, audio_feature_key=audio_feature_key)

    # load from base model
    model = WhisperForConditionalGeneration.from_pretrained(input_arg["model_config"])
    # model.resize_token_embeddings(len(processor.tokenizer))


    model = model.to("cuda")

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    ############
    #  Dataset #
    ############

    if not input_arg.get("load_cache", False):
        # data set
        dataset = load_dataset(
            "csv",
            data_files=input_arg["custom_set_train"],
            cache_dir=input_arg["cache_dir"],
            # cache_dir=None,
        )
        dataset = dataset.filter(lambda e: nlp2.is_file_exist(e["path"]))
        data_train = dataset["train"]
        data_train = data_train.map(
            prepare_dataset_whisper,
            num_proc=1,
            fn_kwargs={"feature_extractor": processor.feature_extractor, "audio_feature_key": audio_feature_key},
        )

        if not input_arg.get("only_eval", False):
            data_train = data_train.map(encode_dataset, fn_kwargs={"processor": processor})
            data_train.save_to_disk(f"{repo_name}/train.data")

        if "custom_set_test" in input_arg:
            dataset_test = load_dataset(
                "csv",
                data_files=input_arg["custom_set_test"],
                cache_dir=input_arg["cache_dir"],
                # cache_dir=None,
            )
            dataset_test = dataset_test.filter(lambda e: nlp2.is_file_exist(e["path"]))
            data_test = dataset_test["train"]
        else:
            dataset = dataset["train"].train_test_split(test_size=0.1)
            data_test = dataset["test"]

        data_test = data_test.map(
            prepare_dataset_whisper,
            num_proc=1,
            fn_kwargs={"feature_extractor": processor.feature_extractor, "audio_feature_key": audio_feature_key},
        )

        data_test = data_test.map(encode_dataset, fn_kwargs={"processor": processor})
        if not input_arg.get("only_eval", False):
            data_test.save_to_disk(f"{repo_name}/test.data")

    else:
        print("Start loading cache dataset")
        data_train = load_from_disk(f"{repo_name}/train.data")
        data_test = load_from_disk(f"{repo_name}/test.data")
    # import json
    # weight = json.load(open("zero_shot.json"))
    # weight = weight["td"] if repo_name.split('/')[1] == "TD_zero_shot" else weight["tat"]
    # total = sum([i[1] for i in weight.items()])
    # for lang, cnt in weight.items():
    #     weight[lang] = cnt/total

    model = experiment(
        input_arg,
        model,
        processor,
        data_collator,
        repo_name,
        data_train,
        data_test,
        time,
        output_dir=input_arg["output_dir"],
        # weight=weight
    )


if __name__ == "__main__":
    main()

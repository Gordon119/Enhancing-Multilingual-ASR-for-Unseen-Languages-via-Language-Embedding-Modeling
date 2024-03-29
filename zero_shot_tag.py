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

class Whisper_Modified(WhisperForConditionalGeneration):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperModel_Modified(config)
        self.proj_out = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.embedding = self.model.decoder.embed_tokens
        self.lang_distribution = None
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        lang_distribution = None
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # print(decoder_inputs_embeds)
        # breakpoint()
        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            lang_distribution=lang_distribution
        )
        lm_logits = self.proj_out(outputs[0])


        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))


        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output


        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor= None,
        stopping_criteria= None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: bool = False,
        return_timestamps: Optional[bool] = None,
        task: Optional[str] = None,
        language: Optional[str] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_condition_type: Optional[str] = None,  # first-segment, all-segments
        condition_on_prev_tokens: Optional[bool] = None,
        temperature: Optional[Union[float, Tuple[float, ...]]] = None,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        num_segment_frames: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: float = 0.02,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        return_dict_in_generate: Optional[bool] = None,
        lang_distribution=None,
        **kwargs,
    ):
        self.lang_distribution = lang_distribution
        # 0. deprecate old inputs
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")
            warnings.warn(
                "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
                FutureWarning,
            )
        # 1. copy generation config
        if generation_config is None:
            generation_config = copy.deepcopy(self.generation_config)
        else:
            generation_config = copy.deepcopy(generation_config)

        # 2. set global generate variables
        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        num_segment_frames = input_stride * self.config.max_source_positions
        batch_size, total_input_frames = self._retrieve_total_input_frames(
            input_features=input_features, input_stride=input_stride, kwargs=kwargs
        )
        is_shortform = total_input_frames <= num_segment_frames

        if is_shortform:
            # warn user of ignored inputs
            self._maybe_warn_unused_inputs(
                condition_on_prev_tokens=condition_on_prev_tokens,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                total_input_frames=total_input_frames,
            )
        # 3. Make sure generation config is correctly set
        # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        self._set_return_outputs(
            return_dict_in_generate=return_dict_in_generate,
            return_token_timestamps=return_token_timestamps,
            is_shortform=is_shortform,
            logprob_threshold=logprob_threshold,
            generation_config=generation_config,
        )
        self._set_return_timestamps(
            return_timestamps=return_timestamps, is_shortform=is_shortform, generation_config=generation_config
        )
        self._set_language_and_task(
            language=language, task=task, is_multilingual=is_multilingual, generation_config=generation_config
        )
        self._set_token_ids(generation_config=generation_config, config=self.config, kwargs=kwargs)
        self._set_num_frames(
            return_token_timestamps=return_token_timestamps, generation_config=generation_config, kwargs=kwargs
        )
        self._set_thresholds_and_condition(
            generation_config=generation_config,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_prev_tokens=condition_on_prev_tokens,
        )
        self._set_prompt_condition_type(
            generation_config=generation_config,
            prompt_condition_type=prompt_condition_type,
        )

        # pass self.config for backward compatibility
        init_tokens = self._retrieve_init_tokens(
            input_features,
            generation_config=generation_config,
            config=self.config,
            num_segment_frames=num_segment_frames,
            kwargs=kwargs,
        )
        # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
        # where the input ids are handled explicitly by the generate method
        self._check_decoder_input_ids(kwargs=kwargs)

        # 3. Retrieve logits processors
        begin_index = len(init_tokens)
        logits_processor = self._retrieve_logit_processors(
            generation_config=generation_config,
            logits_processor=logits_processor,
            begin_index=begin_index,  # begin index is index of first generated decoder token
            is_shortform=is_shortform,
            num_beams=kwargs.get("num_beams", 1),
        )

        # 5. If we're in shortform mode, simple generate the whole input at once and return the output
        if is_shortform:
            if temperature is not None:
                kwargs["temperature"] = temperature

            decoder_input_ids = kwargs.pop("decoder_input_ids", None)
            if decoder_input_ids is None:
                one_tensor = torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
                decoder_input_ids = torch.cat([t * one_tensor for t in init_tokens], dim=-1)

            if prompt_ids is not None:
                decoder_input_ids = torch.cat(
                    [prompt_ids[None].repeat(decoder_input_ids.shape[0], 1), decoder_input_ids], dim=-1
                )

            if kwargs.get("max_new_tokens", 0) + decoder_input_ids.shape[-1] > self.config.max_target_positions:
                max_new_tokens = kwargs.get("max_new_tokens", 0)
                raise ValueError(
                    f"The length of `decoder_input_ids` equal `prompt_ids` plus special start tokens is {decoder_input_ids.shape[-1]}, and the `max_new_tokens` "
                    f"is {max_new_tokens}. Thus, the combined length of "
                    f"`decoder_input_ids` and `max_new_tokens` is: {max_new_tokens + decoder_input_ids.shape[-1]}. This exceeds the "
                    f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                    "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                    f"so that their combined length is less than {self.config.max_target_positions}."
                )
            
            
            outputs = super().generate(
                input_features,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                decoder_input_ids=decoder_input_ids,
                # decoder_inputs_embeds=summation,
                **kwargs,
            )

            if generation_config.return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                outputs["token_timestamps"] = self._extract_token_timestamps(
                    outputs, generation_config.alignment_heads, num_frames=generation_config.num_frames
                )

            return outputs

        # 6. Else we're in longform mode which is more complex.
        # We need to chunk the audio input depending on when the model generates timestamp tokens

        # 6.1 Set and retrieve global longform generation variables
        self._set_condition_on_prev_tokens(
            condition_on_prev_tokens=condition_on_prev_tokens, generation_config=generation_config
        )

        timestamp_begin = generation_config.no_timestamps_token_id + 1
        temperatures = [temperature] if not isinstance(temperature, (list, tuple)) else temperature
        temperature = temperatures[0]
        batch_size = input_features.shape[0]

        max_frames, seek = self._retrieve_max_frames_and_seek(
            batch_size=batch_size, attention_mask=attention_mask, total_input_frames=total_input_frames
        )

        # 6.2 Preppare running variables, list for generation
        cur_bsz = batch_size
        current_segments = self._prepare_segments(
            prompt_ids=prompt_ids,
            batch_size=batch_size,
            generation_config=generation_config,
        )

        batch_idx_map = list(range(batch_size))
        do_condition_on_prev_tokens = [condition_on_prev_tokens for _ in range(batch_size)]

        # 6.2 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            # 6.3 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            input_features, cur_bsz, batch_idx_map = self._maybe_reduce_batch(
                input_features=input_features,
                seek=seek,
                max_frames=max_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )
            time_offset = seek * time_precision / input_stride
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.4 cut out next 30s segment from input features
            segment_input = self._get_input_segment(
                input_features=input_features,
                seek=seek,
                seek_num_frames=seek_num_frames,
                num_segment_frames=num_segment_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )

            # 6.5 prepare decoder input ids
            suppress_tokens = _get_attr_from_logit_processors(
                logits_processor, SuppressTokensLogitsProcessor, "suppress_tokens"
            )
            decoder_input_ids, kwargs = self._prepare_decoder_input_ids(
                cur_bsz=cur_bsz,
                init_tokens=init_tokens,
                current_segments=current_segments,
                batch_idx_map=batch_idx_map,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                prompt_ids=prompt_ids,
                generation_config=generation_config,
                config=self.config,
                device=segment_input.device,
                suppress_tokens=suppress_tokens,
                kwargs=kwargs,
            )

            # 6.6 set max new tokens or max length
            kwargs = self._set_max_new_tokens_and_length(
                config=self.config,
                decoder_input_ids=decoder_input_ids,
                generation_config=generation_config,
                kwargs=kwargs,
            )

            # 6.7 Set current `begin_index` for all logit processors
            for proc in logits_processor:
                if hasattr(proc, "set_begin_index"):
                    proc.set_begin_index(decoder_input_ids.shape[-1])

            # 6.8 Run generate with fallback
            seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens = self.generate_with_fallback(
                segment_input=segment_input,
                decoder_input_ids=decoder_input_ids,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
                seek=seek,
                num_segment_frames=num_segment_frames,
                max_frames=max_frames,
                temperatures=temperatures,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                return_token_timestamps=return_token_timestamps,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                kwargs=kwargs,
            )

            # 6.9 In every generated sequence, split by timestamp tokens and extract segments
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = batch_idx_map[i]

                if should_skip[i]:
                    seek[prev_i] += seek_num_frames[prev_i]
                    continue

                segments, segment_offset = self._retrieve_segment(
                    seek_sequence=seek_sequence,
                    seek_outputs=seek_outputs,
                    time_offset=time_offset,
                    timestamp_begin=timestamp_begin,
                    seek_num_frames=seek_num_frames,
                    time_precision=time_precision,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                    return_token_timestamps=return_token_timestamps,
                )

                current_segments[prev_i] += segments
                seek[prev_i] += segment_offset

        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        final_segments = (
            [x[1:] for x in current_segments]
            if (prompt_ids is not None and generation_config.prompt_condition_type == "first-segment")
            else current_segments
        )
        sequences = _pad_to_max_length(final_segments, generation_config.pad_token_id, padding="right")

        # 8. If we return all segments, the predicted output sequences are put under `"sequences"`.
        if return_segments:
            return {"sequences": sequences, "segments": final_segments}

        return sequences

    def detect_language_custom(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Union[torch.FloatTensor, BaseModelOutput]] = None,
        generation_config: Optional[GenerationConfig] = None,
        num_segment_frames: int = 3000,
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
        non_lang_mask[list(generation_config.lang_to_id.values())] = False

        logits[:, non_lang_mask] = -np.inf

        return logits.softmax(-1)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        # print(decoder_input_ids)
        first = decoder_input_ids.shape[-1] == 3

        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)


        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]


            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1


            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]


            if decoder_position_ids is not None and decoder_position_ids.shape[1] > decoder_input_ids.shape[1]:
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]
        if first:
            template = torch.tensor([[decoder_input_ids[0][0], 0, decoder_input_ids[0][1], decoder_input_ids[0][2]]]).to("cuda")
            summation = torch.zeros((1, 4 ,1280)).to("cuda")
            for i in range(self.lang_distribution.shape[0]):
                value = self.lang_distribution[i]
                if value.item() == 0:
                    continue
                template[0][1] = i
                summation += self.lang_distribution[i] * self.embedding(template)
            return {
                "encoder_outputs": encoder_outputs,
                "past_key_values": past_key_values,
                # "decoder_input_ids": decoder_input_ids,
                "decoder_inputs_embeds": summation,
                "use_cache": use_cache,
                "decoder_attention_mask": decoder_attention_mask,
                "decoder_position_ids": decoder_position_ids,
            }
        else:
            return {
                "encoder_outputs": encoder_outputs,
                "past_key_values": past_key_values,
                "decoder_input_ids": decoder_input_ids,
                "use_cache": use_cache,
                "decoder_attention_mask": decoder_attention_mask,
                "decoder_position_ids": decoder_position_ids,
            }

class WhisperModel_Modified(WhisperModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder_Modified(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        lang_distribution = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)


            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        if not return_dict:
            return decoder_outputs + encoder_outputs


        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

import math
class WhisperDecoder_Modified(WhisperDecoder):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0


        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        # self.embed_tokens = embedding
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)


        self.layers = torch.nn.ModuleList([WhisperDecoderLayer(config) for _ in range(config.decoder_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"


        self.layer_norm = torch.nn.LayerNorm(config.d_model)


        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            # print("decoder:",input_ids)
            inputs_embeds = self.embed_tokens(input_ids)
            # print("decoder:",inputs_embeds)
        else:
            print("success")


        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and head_mask is None and not output_attentions:
            # output_attentions=True & head_mask can not be supported when using SDPA.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(
                input_ids, past_key_values_length=past_key_values_length, position_ids=position_ids
            )
        else:
            positions = self.embed_positions(
                inputs_embeds, past_key_values_length=past_key_values_length, position_ids=position_ids
            )

        hidden_states = inputs_embeds + positions
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,  # past_key_value
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

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
                    decoder_input_ids=batch["labels"][:, :3].to("cuda"),
                    max_new_tokens=255,
                    lang_distribution=model.detect_language_custom(input_features=batch["input_features"].to("cuda")).squeeze(),
                    task="transcribe"
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
        , language=None
    )
    # special_tokens_dict = {'additional_special_tokens': ['<|tat|>', '<|td|>'] + processor.tokenizer.all_special_tokens}
    # special_tokens_dict = {'additional_special_tokens': ['<|tat|>'] + processor.tokenizer.all_special_tokens}
    # num_added_toks = processor.tokenizer.add_special_tokens(special_tokens_dict)
    processor.save_pretrained(repo_name)
    audio_feature_key = "input_ids"
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, audio_feature_key=audio_feature_key)

    # load from base model
    model = Whisper_Modified.from_pretrained(input_arg["model_config"])
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

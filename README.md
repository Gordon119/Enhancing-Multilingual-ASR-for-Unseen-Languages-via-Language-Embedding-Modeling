# Enhancing Multilingual ASR for Unseen Languages via Language Embedding Modeling (IEEE ICASSP 2025)
Paper: https://ieeexplore.ieee.org/document/10890363
## Installation
Use the exact version of the packages in the ``requirements.txt``.
```
pip install -r requirements.txt
```

## Data Format
The data format follows ml_superb. Each instance contains the path to the wav file and the text.
For example,
```
path,text
data/ml_superb/sixth_edition/fleurs/ast/wav/fleurs_ast_000067.wav,EN CUANTES A XAPóN XAPóN YERA UN PAíS-ISLLA IGUAL QUE GRAN BRETAñA
data/ml_superb/sixth_edition/fleurs/ast/wav/fleurs_ast_000068.wav,DE FRACASAR LOS ALIAOS YE PROBABLE QU'ALEMAñA CONQUISTARE GRAN BRETAñA Y EL RESTU D'EUROPA
data/ml_superb/sixth_edition/fleurs/ast/wav/fleurs_ast_000069.wav,LES IMáXENES D’INFRARROXU AMUESEN QUE LES VARIACIONES DE TEMPERATURA ENTE’L DíA Y LA NUECHE PRUEBEN QUE YE FáCIL QUE SEYAN CUEVES
...
```
You can use the scripts ``gen_data_seen.py`` and ``gen_data.py`` in ``tools`` to generate Whisper-seen and Whisper-unseen data.

## Zero-shot
There are three settings for this experiment: Vanilla, Utterance-wise Weighted Sum and Corpus-wise Weighted Sum.
For the weighted sum methods, please refer to ``ws_zero_shot.sh`` for the usage of ``ws_zero_shot.py``.
For the vanilla method, please use the script ``vanilla_zero_shot.sh``

## Finetuning
For the finetuning experiments, there are two additional methods: Trainable Weighted Sum and Predictor Based methods.
The following are the scripts and their corresponding settings:

- Vanilla: ``vanilla_finetune.sh``
- Utterance-wise and Corpus-wise Weighted Sum: ``ws_finetune_untrainalbe.sh``
- Trainable Weighted Sum: ``ws_finetune_trainable.sh``

As for the Predictor-base method, a mlp must be trained first to get the predictor.

- Get training data (weight-embedding pairs) and train the predictor: ``get_predictor.sh``
- Use the Predictor: ``utterance_wise_with_predictor.sh`` and ``corpus_wise_with_predictor.sh``

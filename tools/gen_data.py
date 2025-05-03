import glob, os
import pandas as pd
from iso639 import Lang

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
for y, z in [("train", "transcript_1h_train"), ("val", "transcript_10min_dev"), ("test", "transcript_10min_test")]:
    files = glob.glob(f"Whisper_Experiments/data/ml_superb/sixth_edition/*/*/{z}.txt")
    dfs = {}
    res = []
    langs = []
    skip = set()
    for key, val in LANGUAGES.items():
        langs.append(val)
    # print(langs)
    for file in files:
        try:
            lang_code = Lang(os.path.basename(os.path.dirname(file))).name.lower()
        except:
            continue
        if lang_code in langs:
            skip.add(lang_code)
            continue
        lang_code = os.path.basename(os.path.dirname(file))
        try:
            df = pd.read_csv(file, sep="\t", usecols =[0, 2], names = ["path", "text"])
        except pd.errors.ParserError:
            try:
                df = pd.read_csv(file, sep="\t", usecols =[0, 1], names = ["path", "text"])
            except pd.errors.ParserError:
                continue
        for i in range(len(df)):
            df.loc[i, "path"] = os.path.join(os.path.dirname(file), f"wav/{df.loc[i, 'path']}.wav")
        if dfs.get(lang_code, False):
            dfs[lang_code].append(df)
        else:
            dfs[lang_code] = [df]
        from pathlib import Path
        Path(lang_code).mkdir(parents=True, exist_ok=True)
    for key, val in dfs.items():
        if len(val) > 1:
            pd.concat(val).to_csv(f"{key}/{y}.csv", index=False)
        else:
            val[0].to_csv(f"{key}/{y}.csv", index=False)
    print(dfs.keys())
    # print(skip)

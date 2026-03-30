import os
import pickle
import random
from pathlib import Path

import datasets
import transformers

# Bundled repo root (FlatQuantBundled/). Local ./datasets/* lives here in upstream FlatQuant;
# when missing, fall back to Hugging Face Hub so calibration works from any cwd (e.g. agent outputs/).
_FLATQUANT_ROOT = Path(__file__).resolve().parents[1]


def _wikitext_local_dir() -> Path | None:
    p = _FLATQUANT_ROOT / "datasets" / "wikitext"
    return p if p.is_dir() else None


def _load_wikitext_raw_v1(split: str):
    local = _wikitext_local_dir()
    if local is not None:
        return datasets.load_dataset(str(local), "wikitext-2-raw-v1", split=split)
    return datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = _load_wikitext_raw_v1("test")
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = _load_wikitext_raw_v1("train")
        traindata = traindata.filter(lambda x: len(x) > 0)
        traindata = traindata.map(lambda x : {'text': x['text'].strip()})
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seqlen, tokenizer, eval_mode=False):
    c4_root = _FLATQUANT_ROOT / "datasets" / "allenai" / "c4"
    if not c4_root.is_dir():
        raise FileNotFoundError(
            f"C4 calibration expects vendored data at {c4_root} (upstream FlatQuant layout). "
            "Use --cali_dataset wikitext2, or clone the FlatQuant datasets/ tree into FlatQuantBundled/datasets."
        )
    if eval_mode:
        valdata = datasets.load_dataset(
            str(c4_root),
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation',
        )
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            str(c4_root),
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train',
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=False):
    ptb_root = _FLATQUANT_ROOT / "datasets" / "ptb_text_only"
    if not ptb_root.is_dir():
        raise FileNotFoundError(
            f"PTB expects vendored data at {ptb_root}. Use --cali_dataset wikitext2 or add datasets/ptb_text_only."
        )
    if eval_mode:
        testdata = datasets.load_dataset(str(ptb_root), 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset(str(ptb_root), 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_pile(nsamples, seqlen, tokenizer):
    pile_root = _FLATQUANT_ROOT / "datasets" / "pile-val-backup"
    if not pile_root.is_dir():
        raise FileNotFoundError(
            f"Pile calibration expects vendored data at {pile_root}. "
            "Use --cali_dataset wikitext2 or add datasets/pile-val-backup."
        )
    traindata = datasets.load_dataset(str(pile_root), split="validation")
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_loaders(
    args, name, tokenizer, nsamples=128, seqlen=2048, eval_mode=False
):
    if 'wikitext2' in name:
        dataset = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode)
    elif 'ptb' in name:
        dataset = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'c4' in name:
        dataset = get_c4_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'pile' in name:
        dataset = get_pile(nsamples, seqlen, tokenizer)

    if 'c4' in name and eval_mode:
        dataset = dataset.input_ids
        dataset = TokenizerWrapper(dataset)
    return dataset

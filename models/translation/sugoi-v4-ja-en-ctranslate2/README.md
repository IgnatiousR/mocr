---
license: other
license_name: ntt-license
license_link: LICENSE
language:
- ja
- en
pipeline_tag: translation
library_name: fairseq
tags:
- nmt
---

# Sugoi v4 JPN->ENG NMT Model by MingShiba

- https://sugoitoolkit.com
- https://blog.sugoitoolkit.com
- https://www.patreon.com/mingshiba

## How to download this model using python
- Install Python https://www.python.org/downloads/
- `cmd`
- `python --version`
- `python -m pip install huggingface_hub`
- `python`

```
import huggingface_hub
huggingface_hub.download_snapshot('entai2965/sugoi-v4-ja-en-ctranslate2',local_dir='sugoi-v4-ja-en-ctranslate2')
```

## How to run this model (batch syntax)
- https://opennmt.net/CTranslate2/guides/fairseq.html#fairseq
- `cmd`
- `python -m pip install ctranslate2 sentencepiece`
- `python`
```
import ctranslate2
import sentencepiece

#set defaults
model_path='sugoi-v4-ja-en-ctranslate2'
sentencepiece_model_path=model_path+'/spm'

device='cpu'
#device='cuda'

#load data
string1='は静かに前へと歩み出た。'
string2='悲しいGPTと話したことがありますか?'
raw_list=[string1,string2]

#load models
translator = ctranslate2.Translator(model_path, device=device)
tokenizer_for_source_language = sentencepiece.SentencePieceProcessor(sentencepiece_model_path+'/spm.ja.nopretok.model')
tokenizer_for_target_language = sentencepiece.SentencePieceProcessor(sentencepiece_model_path+'/spm.en.nopretok.model')

#tokenize batch
tokenized_batch=[]
for text in raw_list:
    tokenized_batch.append(tokenizer_for_source_language.encode(text,out_type=str))

#translate
#https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html?#ctranslate2.Translator.translate_batch
translated_batch=translator.translate_batch(source=tokenized_batch,beam_size=5)
assert(len(raw_list)==len(translated_batch))

#decode
for count,tokens in enumerate(translated_batch):
    translated_batch[count]=tokenizer_for_target_language.decode(tokens.hypotheses[0]).replace('<unk>','')

#output
for text in translated_batch:
    print(text)
```

[Functional programming](https://docs.python.org/3/howto/functional.html) version

```
import ctranslate2
import sentencepiece

#set defaults
model_path='sugoi-v4-ja-en-ctranslate2'
sentencepiece_model_path=model_path+'/spm'

device='cpu'
#device='cuda'

#load data
string1='は静かに前へと歩み出た。'
string2='悲しいGPTと話したことがありますか?'
raw_list=[string1,string2]

#load models
translator = ctranslate2.Translator(model_path, device=device)
tokenizer_for_source_language = sentencepiece.SentencePieceProcessor(sentencepiece_model_path+'/spm.ja.nopretok.model')
tokenizer_for_target_language = sentencepiece.SentencePieceProcessor(sentencepiece_model_path+'/spm.en.nopretok.model')

#invoke black magic
translated_batch=[tokenizer_for_target_language.decode(tokens.hypotheses[0]).replace('<unk>','') for tokens in translator.translate_batch(source=[tokenizer_for_source_language.encode(text,out_type=str) for text in raw_list],beam_size=5)]
assert(len(raw_list)==len(translated_batch))

#output
for text in translated_batch:
    print(text)
```

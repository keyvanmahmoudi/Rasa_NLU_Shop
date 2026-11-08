# Rasa NLU for Persian Shopping Dialogues

A natural language understanding (NLU) model for Persian customer–seller conversations in a supermarket, built with [Rasa](https://rasa.com/) 3. Given a customer utterance such as «شیر کم‌چرب می‌خوام» ("I want low-fat milk"), the model predicts the **intent** (here, `purchase`) and extracts the **entities** (product, fat content, brand, volume, ...). The data covers two product domains: dairy and drinks.

This is an early experiment (2023) in my research on Persian task-oriented dialogue. The later, larger work is [**PerSHOP**](https://github.com/keyvanmahmoudi/PerSHOP), a dataset of 1,061 annotated Persian shopping dialogues across 14 product domains.

## Background

A task-oriented dialogue system is classically built from four components: natural language understanding (NLU), dialogue state tracking, dialogue policy, and natural language generation. This repository covers the first one: understanding what the user wants and which product attributes they mention.

## Model

The pipeline ([`config.yml`](config.yml), `language: fa`):

* `WhitespaceTokenizer`
* `LanguageModelFeaturizer` with multilingual [LaBSE](https://huggingface.co/rasa/LaBSE) embeddings
* `LexicalSyntacticFeaturizer`, `RegexFeaturizer`, and word- and character-level (1–4-gram) `CountVectorsFeaturizer`
* `DIETClassifier` (joint intent classification and entity recognition, 100 epochs)
* `RegexEntityExtractor` with a brand lookup table, and `EntitySynonymMapper` for spelling variants of product and brand names
* `FallbackClassifier` (threshold 0.3)

**Intents (12):** `purchase`, `inform`, `prev_inform`, `ask_detail`, `user_ask`, `User_Confirm`, `Deny`, `Regret`, `don’t_care`, `Greeting`, `Goodby`, `chitchat`

**Entities (18):** `product`, `brand`, `type`, `taste`, `size`, `weight`, `volume`, `cat`, `charbi` (fat content), `gazdar` (carbonated), `package`, `bag`, `bag_number`, `family_drink`, `cheese_shape`, `doogh_sabzi`, `tea_aromatic`, `refresh`

## Results

Evaluated on a held-out test set of 151 utterances ([`train_test_split/`](train_test_split)). Full reports, confusion matrices and error files are in [`results-lapse/`](results-lapse).

| Task | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Intent classification | 82.8% | 0.83 | 0.65 |
| Entity recognition (DIET) | — | 0.78 | 0.74 |

The most frequent intent, `purchase` (87 test utterances), reaches F1 = 0.90. Among entities, `weight` (F1 = 0.95) and `volume` (0.91) are recognized best; `brand`, the most frequent entity (113 test mentions), reaches 0.77. Rare intents with only two or three test examples account for most of the gap between weighted and macro F1.

## Repository structure

| Path | Contents |
|---|---|
| `config.yml` | NLU pipeline and dialogue policies |
| `domain.yml` | Intents, entities, slots and responses |
| `data/nlu.yml` | Lookup tables, synonyms and regular expressions |
| `train_test_split/` | Training and test NLU data |
| `results-lapse/` | Evaluation reports and confusion matrices |
| `actions/`, `data/stories.yml`, `data/rules.yml` | Rasa project scaffolding (dialogue management is not part of this experiment) |

## Usage

```bash
pip install rasa==3.6.*
rasa train nlu --nlu train_test_split/training_data.yml
rasa test nlu --nlu train_test_split/test_data.yml
rasa shell nlu
```

## Author

**Keyvan Mahmoudi**, PhD student in Natural Language Processing, University of Tehran
[LinkedIn](https://www.linkedin.com/in/keyvan-mahmoudi-7b3280103) · [Hugging Face](https://huggingface.co/KeyvanMahmoudi)

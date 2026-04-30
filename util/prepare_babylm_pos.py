import spacy
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
DATASET_NAME = "BabyLM-community/BabyLM-2026-Strict-Small"
TOKENIZER_NAME = "gpt2"
OUTPUT_DIR = "./babylm_contextual_tagged"
BATCH_SIZE = 1000  # Number of rows to process at once
NUM_CPUS = 4

# UPOS Tags mapped to integers
UPOS_MAP = {
    "ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5,
    "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11,
    "PUNCT": 12, "SCONJ": 13, "SYM": 14, "VERB": 15, "X": 16
}


def prepare_babylm_pos():
    print(f"Loading tokenizer '{TOKENIZER_NAME}' and spaCy...")
    # Use the fast tokenizer to get character offsets
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    # Disable components we don't need for pure POS tagging to massively speed up processing
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])

    print(f"Downloading dataset '{DATASET_NAME}'...")
    # This downloads and caches the dataset locally
    dataset = load_dataset(DATASET_NAME)

    def add_contextual_tags(examples):
        texts = examples["text"]

        # 1. Tokenize with Hugging Face (get subwords and their character offsets)
        # We don't pad here; we just want the raw token streams for language modeling
        tokenized_inputs = tokenizer(
            texts,
            truncation=True,
            max_length=1024,
            return_offsets_mapping=True,
            return_attention_mask=False
        )

        # 2. Tokenize the exact same texts with spaCy
        # nlp.pipe is highly optimized for processing batches of text
        spacy_docs = list(nlp.pipe(texts))

        all_pos_tags = []

        # 3. Align the subwords to the spaCy whole words
        for doc, offsets in zip(spacy_docs, tokenized_inputs["offset_mapping"]):
            pos_ids = []

            for start_char, end_char in offsets:
                # Special tokens (like sequence start/end) have offset (0,0)
                if start_char == end_char:
                    pos_ids.append(UPOS_MAP["X"])
                    continue

                # Find which spaCy token covers the start character of this subword
                assigned_pos = UPOS_MAP["X"]
                for token in doc:
                    token_start = token.idx
                    token_end = token.idx + len(token.text)

                    # If the subword's start character falls inside this spaCy token
                    if token_start <= start_char < token_end:
                        assigned_pos = UPOS_MAP.get(token.pos_, UPOS_MAP["X"])
                        break

                pos_ids.append(assigned_pos)

            all_pos_tags.append(pos_ids)

        # Add our new array to the Hugging Face dataset dictionary
        tokenized_inputs["pos_tags"] = all_pos_tags

        # We must remove the offset_mapping because HF Datasets cannot easily save tuples of varying lengths
        del tokenized_inputs["offset_mapping"]

        return tokenized_inputs

    print("Processing and aligning dataset (this may take a while)...")
    # map() applies our function to the dataset. batched=True makes it efficient.
    # remove_columns strips the original raw text to save space.
    processed_dataset = dataset.map(
        add_contextual_tags,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_CPUS,
        remove_columns=dataset["train"].column_names,
        desc="Aligning POS tags to BPE tokens"
    )

    print(f"Saving processed dataset to '{OUTPUT_DIR}'...")
    processed_dataset.save_to_disk(OUTPUT_DIR)
    print("Complete! Ready for training.")


def build_static_vocab_map():
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    vocab = tokenizer.get_vocab()  # Dictionary of {string: id}
    vocab_size = tokenizer.vocab_size

    # Initialize map with 'X' (Unknown)
    vocab_pos_map = torch.full((vocab_size,), UPOS_MAP["X"], dtype=torch.long)

    print(f"Mapping {vocab_size} tokens to POS tags...")

    for token_str, token_id in tqdm(vocab.items()):
        # GPT-2 cleanup: 'Ġ' represents a space prefix
        clean_token = token_str.replace('Ġ', '').strip()

        if not clean_token:
            continue

        # Heuristic for punctuation
        if not any(char.isalnum() for char in clean_token):
            vocab_pos_map[token_id] = UPOS_MAP["PUNCT"]
            continue

        # Use spaCy to predict the tag for the isolated token
        doc = nlp(clean_token)
        if len(doc) > 0:
            pos_tag = doc[0].pos_
            vocab_pos_map[token_id] = UPOS_MAP.get(pos_tag, UPOS_MAP["X"])

    # Save the tensor
    torch.save(vocab_pos_map, "vocab_pos_map.pt")
    print("Success! 'vocab_pos_map.pt' created.")


if __name__ == "__main__":
    build_static_vocab_map()
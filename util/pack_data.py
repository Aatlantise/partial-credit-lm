from datasets import load_from_disk
import multiprocessing

# 1. Load your already-tagged dataset
dataset = load_from_disk("./babylm_contextual_tagged")
block_size = 1024


def pack_sequences(examples):
    # Flatten the lists of lists into a single massive list
    concatenated_ids = sum(examples["input_ids"], [])
    concatenated_pos = sum(examples["pos_tags"], [])

    # Find out how many clean 1024-token blocks we can make
    total_length = len(concatenated_ids)
    total_length = (total_length // block_size) * block_size

    # Chop them up into perfect 1024 blocks
    result = {
        "input_ids": [
            concatenated_ids[i: i + block_size]
            for i in range(0, total_length, block_size)
        ],
        "pos_tags": [
            concatenated_pos[i: i + block_size]
            for i in range(0, total_length, block_size)
        ],
    }
    return result


print("Packing sequences into 1024-token blocks...")
num_cpus = multiprocessing.cpu_count()

# Apply the packing
packed_dataset = dataset.map(
    pack_sequences,
    batched=True,
    batch_size=1000,
    num_proc=num_cpus,
    remove_columns=dataset["train"].column_names,
    desc="Packing dataset"
)

print(f"Original rows: {len(dataset['train'])}")
print(f"Packed rows: {len(packed_dataset['train'])}")

packed_dataset.save_to_disk("./babylm_packed_1024")
print("Done! Point your training script to ./babylm_packed_1024")
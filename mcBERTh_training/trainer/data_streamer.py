"""
This script builds a streaming dataset with decade-balancing and shuffling using Hugging face datasets.

Streaming, shuffled & interleaved;

data directory layout:
  coha_sharded_full/
    train/<decade>/shard_*.jsonl
    valid/<decade>/shard_*.jsonl
    
    (Only one shard per decade.)

Each JSONL line in our dataset includes the following fields: 

    {"file_name": "...", "doc_id": "...",  "decade": "...", 
    "genre": "...",  chunk_id: "...", "text": "...",}.

"""

from datasets import interleave_datasets, load_dataset, IterableDataset
from google_cloud_save import gcs_get_dataset_json_data


DECADES = [
    "1810s", "1820s", "1830s", "1840s", "1850s", "1860s", "1870s", "1880s", "1890s", "1900s",
    "1910s", "1920s", "1930s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s"
]


def gen(data):
    for item in data:
        text = item['text']  # type: ignore
        date = item['decade']  # type: ignore
        date = str(date).removesuffix('s')
        yield {
            'decade': date,
            'text': f'<decade_{date}> ' + text
        }


def build_decade_balanced_stream(
        service_account_path=None,
        root="coha_sharded_full",
        split="train",
        buffer_size=50_000,
        seed=123,
        probabilities=None,
        stopping_strategy="all_exhausted",):
    """     
    Return: 
        A mixed IterableDataset, a streaming iterable. 
    """
    # hugging face datasets:
    # Stream and interleave datasets: https://huggingface.co/docs/datasets/stream#interleave-datasets

    # a list of streaming datasets
    streams = []

    for decade in DECADES:
        # one streaming dataset per decade (streaming=True)
        if split == 'valid' and decade == '1810s':
            continue
        raw_data = gcs_get_dataset_json_data(
            credentials_path=service_account_path, bucket_name="project3102-data-bucket", data_blob_path=f"{root}/{split}/{decade}/shard_000.jsonl")
        dataset = IterableDataset.from_generator(generator=gen, gen_kwargs={
                                                 # pyright: ignore[reportArgumentType]
                                                 'data': raw_data})
        # dataset = load_dataset(
        #     "json",
        #     data_files=[f"{root}/{split}/{decade}/shard_*.jsonl"],
        #     split="train",
        #     streaming=True,
        # )
        # shuffle within decade
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
        streams.append(dataset)

    # interleave decades
    mixed = interleave_datasets(streams, probabilities=probabilities,
                                seed=seed, stopping_strategy=stopping_strategy)  # type: ignore

    # shuffle the mixed dataset
    mixed = mixed.shuffle(buffer_size=buffer_size, seed=seed)

    return mixed


def main():

    my_dataset = build_decade_balanced_stream(
        "nlp-research-sp26-8499634f1c62.json")

    # Take a look at the first 32 examples from the dataset stream
    for i, example in enumerate(my_dataset):
        if i >= 2:
            break
        print(example)

    print(len(my_dataset))


if __name__ == "__main__":
    main()

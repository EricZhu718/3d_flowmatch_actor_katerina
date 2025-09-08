from .rlbench import (
    Peract2Dataset,
    Peract2SingleCamDataset,
    PeractDataset,
    PeractTwoCamDataset,
    HiveformerDataset,
    EgoDexDataset
)


def fetch_dataset_class(dataset_name):
    """Fetch the dataset class based on the dataset name."""
    dataset_classes = {
        "Peract2_3dfront_3dwrist": Peract2Dataset,
        "Peract2_3dfront": Peract2SingleCamDataset,
        "Peract": PeractDataset,
        "PeractTwoCam": PeractTwoCamDataset,
        "HiveformerRLBench": HiveformerDataset,
        "CustomEgoDexDataset" : EgoDexDataset
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_classes[dataset_name]

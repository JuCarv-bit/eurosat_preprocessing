import torch
from torchvision import transforms

MS_PATH  = "/users/c/carvalhj/datasets/eurosat/EuroSAT_MS/"
RGB_PATH = "/users/c/carvalhj/datasets/eurosat/EuroSAT_RGB/"
BATCH_SIZE = 4

from information_extraction import (
    extract_metadata,
    EuroSATDataset,
    get_data_loaders,
)

def test_extract_metadata():
    df = extract_metadata(MS_PATH, RGB_PATH)
    assert len(df) > 0, "No metadata records found!"
    for col in ["filepath", "latitude", "longitude", "label_id"]:
        assert col in df.columns, f"Missing column: {col}"
    print("✓ extract_metadata OK")
    print(df.head())

def test_dataset_getitem():
    df = extract_metadata(MS_PATH, RGB_PATH)
    ds = EuroSATDataset(RGB_PATH, df, transform=transforms.ToTensor())
    img, meta, label = ds[0]
    assert isinstance(img, torch.Tensor),     "Image is not a Tensor"
    assert img.shape[0] == 3,                 "Image does not have 3 channels"
    assert isinstance(meta, torch.Tensor),    "Metadata is not a Tensor"
    assert meta.shape == (2,),                "Metadata tensor should be shape (2,)"
    print("label type:", type(label))
    assert isinstance(label, int),            "Label is not an int"
    print("✓ EuroSATDataset.__getitem__ OK")

def test_get_data_loaders():
    loaders = get_data_loaders(MS_PATH, RGB_PATH, batch_size=BATCH_SIZE)
    train_loader, val_loader, test_loader, val_no_transform_ds, num_classes = loaders
    
    train_data = next(iter(train_loader))
    imgs, imgs_2, metas = train_data # train is just simclr augmentation + metadata
    test_data = next(iter(test_loader))
    img_test, metas_test, labels_test = test_data
    val_data = next(iter(val_loader))
    img_val, metas_val, labels_val = val_data

    print("imgs shape:", imgs.shape)
    print("metas shape:", metas.shape)
    print("labels shape:", labels_test.shape)
    print("batch size:", BATCH_SIZE)
    print( metas.shape[1], "should be 2 (lat, lon)")
    assert imgs.shape[0] == BATCH_SIZE,      "Wrong batch size for images"
    assert imgs.shape[1] == 3,               "Images should have 3 channels"
    assert metas.shape[0] == BATCH_SIZE      , "Wrong batch size for metadata"
    assert metas.shape[1] == 2,              "Metadata should have 2 values (lat, lon)"
    assert labels_test.shape[0] == BATCH_SIZE,    "Labels batch size mismatch"
    assert isinstance(num_classes, int) and num_classes > 0
    print("get_data_loaders OK (train loader batch dims & num_classes)")

if __name__ == "__main__":
    test_extract_metadata()
    test_dataset_getitem()
    test_get_data_loaders()
    print("All tests passed!")  

import tensorflow as tf
import tensorflow_datasets as tfds

def load_tfds_pipeline(
    batch_size: int = 32,
    image_size: tuple[int, int] = (128, 128),
    split: str = "train",
    shuffle_buffer_size: int = 1000
) -> tf.data.Dataset:
    """
    Creates an efficient data pipeline using the PlantVillage dataset from TFDS.
    
    Args:
        batch_size: Number of samples per batch.
        image_size: Target image size (width, height) for resizing.
        split: The dataset split to load (e.g., 'train').
        shuffle_buffer_size: Buffer size for shuffling files.
        
    Returns:
        A ready-to-use tf.data.Dataset object compatible with model.fit().
    """
    # 1. Load dataset metadata and stream from TFDS
    # Note: 'plant_village' dataset typically has only a 'train' split by default.
    dataset, info = tfds.load(
        name="plant_village",
        split=split,
        with_info=True,
        as_supervised=True,  # Returns (image, label) tuples
        shuffle_files=True    # Shuffle raw files for better randomness
    )
    
    # 2. Preprocessing function
    def preprocess_image(image, label):
        # Resize to desired dimensions
        image = tf.image.resize(image, image_size)
        
        # Cast to float32 (keep in 0-255 range for preprocess_input layer)
        image = tf.cast(image, tf.float32)
        
        return image, label

    # 3. Optimize the data pipeline
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if split == "train":
        # Shuffle for optimal stochasticity
        dataset = dataset.shuffle(shuffle_buffer_size)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, info

if __name__ == "__main__":
    # Test the pipeline
    train_dataset, ds_info = load_tfds_pipeline(batch_size=32, image_size=(128, 128), split="train")
    
    print(f"Number of classes: {ds_info.features['label'].num_classes}")
    print(f"Class names: {ds_info.features['label'].names[:5]}...")
    
    for images, labels in train_dataset.take(1):
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Min pixel value: {tf.reduce_min(images).numpy()}")
        print(f"Max pixel value: {tf.reduce_max(images).numpy()}")

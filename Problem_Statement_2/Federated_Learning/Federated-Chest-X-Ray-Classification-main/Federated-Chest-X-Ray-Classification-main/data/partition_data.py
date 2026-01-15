import os, shutil, random
from sklearn.model_selection import train_test_split

RAW_DIR = 'data/raw/chest_xray'
PARTITIONS_DIR = 'data/partitions'
NUM_CLIENTS = 3
TEST_SIZE = 0.2  # 20% for test

def get_all_samples():
    samples = []
    for split in ['train', 'val', 'test']:
        for label in ['NORMAL', 'PNEUMONIA']:
            d = os.path.join(RAW_DIR, split, label)
            for img in os.listdir(d):
                samples.append((os.path.join(d, img), label))
    random.shuffle(samples)
    return samples

def partition_and_split():
    samples = get_all_samples()
    partitions = [[] for _ in range(NUM_CLIENTS)]
    for idx, s in enumerate(samples):
        partitions[idx % NUM_CLIENTS].append(s)
    for i, client_samples in enumerate(partitions):
        train_samples, test_samples = train_test_split(client_samples, test_size=TEST_SIZE, random_state=42, stratify=[label for _, label in client_samples])
        for split_name, split_samples in [('train', train_samples), ('test', test_samples)]:
            for img_path, label in split_samples:
                dest_dir = os.path.join(PARTITIONS_DIR, f'client_{i+1}', split_name, label)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(img_path, dest_dir)
    print("Done partitioning and splitting!")

if __name__ == '__main__':
    partition_and_split()
import pandas as pd
from pathlib import Path


def convert_data_set(data_path):
    df = pd.read_csv(data_path, header=None)
    df.columns = ['image_path']
    df['label'] = df['image_path'].apply(lambda x: 1 if 'positive' in x else 0)
    return df


if __name__ == '__main__':
    dir_path = Path('data/MURA-v1.1/')

    train_set = convert_data_set(dir_path / 'train_image_paths.csv')

    train_positive_set = train_set[train_set['label'] == 1]
    train_negative_set = train_set[train_set['label'] == 0]

    train_positive_set.to_csv(
        dir_path / 'train_positive_data.csv', index=False)
    train_negative_set.to_csv(
        dir_path / 'train_negative_data.csv', index=False)

    val_set = convert_data_set(dir_path / 'valid_image_paths.csv')
    val_set.to_csv(dir_path / 'val_data.csv', index=False)

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


def transform(train=False):
    # Resize the whole lesion image, then only a mild scale crop in training.
    ops = [T.Resize((224, 224))]
    if train:
        ops += [T.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class ISICDataset(Dataset):
    def __init__(self, rows, root, train=False):
        self.rows, self.root, self.transform = rows, Path(root), transform(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as im:
            x = self.transform(im.convert('RGB'))
        return x, row['label'], row['image_id']

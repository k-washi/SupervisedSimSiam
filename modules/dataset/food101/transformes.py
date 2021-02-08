import albumentations as A
from albumentations.augmentations import transforms
from albumentations.augmentations.transforms import RandomSizedCrop
from albumentations.pytorch import ToTensor

ColorEffectP=0.5
def generate_train_transformes(target_size=(50, 50)):
    transforms = [
        A.Resize(height=target_size[0], width=target_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.25),
        A.Rotate(limit=(-90,90),p=1.),
        A.RandomResizedCrop(height=target_size[0], width=target_size[0],scale=(0.5, 1.), p=0.5),
        A.OneOf([
            A.Blur(p=ColorEffectP),
            A.GaussNoise(p=ColorEffectP),
            A.Downscale(p=ColorEffectP),
            A.RGBShift(p=ColorEffectP, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            A.RandomGamma(p=ColorEffectP),
            A.RandomBrightnessContrast(p=ColorEffectP)
        ],p = 0.5),
        ToTensor()
    ]
    return A.Compose(transforms)

def generate_test_transforms(target_size=(50, 50)):
    transforms = [
        A.Resize(height=target_size[0], width=target_size[1]),
        ToTensor()
    ]
    return A.Compose(transforms)
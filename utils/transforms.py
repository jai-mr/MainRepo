import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2


def apply_transforms(mean, std):
    """
    Image augmentations for train and test set.
    """
    train_transforms = A.Compose(
        [
            A.Sequential(
                [
                    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                    A.RandomCrop(width=32, height=32, p=1),
                ],
                p=0.5,
            ),
            A.Rotate(limit=5, p=0.2),
            A.CoarseDropout(
                max_holes=1,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=tuple((x * 255.0 for x in mean)),
                p=0.2,
            ),
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    return (
        lambda img: train_transforms(image=np.array(img))["image"],
        lambda img: test_transforms(image=np.array(img))["image"],
    )


def apply_transforms_custom_resnet(mean, std):
    """
    Image augmentations for train and test set.
    """
    train_transforms = A.Compose([A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                                  A.RandomCrop(width=32, height=32,p=4),
                                  A.Rotate(limit=5),
                                  #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.25),
                                #   A.CoarseDropout(max_holes=1,min_holes = 1, max_height=16, max_width=16, p=0.5,fill_value=tuple([x * 255.0 for x in DATA_MEAN]),
                                #   min_height=16, min_width=16),
                                  A.HorizontalFlip(p=0.3),
                                  A.VerticalFlip(p=0.1),
                                  A.RandomBrightnessContrast(p=0.2),
                                  A.Normalize(mean=mean, std=std,always_apply=True),
                                  A.Cutout(num_holes=1,max_h_size=16, max_w_size=16),
                                  ToTensorV2()
                                ])    

    test_transforms = A.Compose(
        [
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    return (
        lambda img: train_transforms(image=np.array(img))["image"],
        lambda img: test_transforms(image=np.array(img))["image"],
    )


def apply_transforms_tiny_imagenet(mean, std):
    """
    Image augmentations for train and test set for Tiny ImageNet.
    """
    train_transforms = A.Compose([A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                              A.RandomCrop(width=32, height=32,p=4),
                              A.Rotate(limit=5),
                              #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.25),
                            #   A.CoarseDropout(max_holes=1,min_holes = 1, max_height=16, max_width=16, p=0.5,fill_value=tuple([x * 255.0 for x in DATA_MEAN]),
                            #   min_height=16, min_width=16),
                              A.HorizontalFlip(p=0.3),
                              A.VerticalFlip(p=0.1),
                              A.RandomBrightnessContrast(p=0.2),
                              A.Normalize(mean=mean, std=std,always_apply=True),
                              A.Cutout(num_holes=1,max_h_size=16, max_w_size=16),
                              ToTensorV2()
                            ])    


    test_transforms = A.Compose(
        [
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),
        ]
    )

    return (
        lambda img: train_transforms(image=np.array(img))["image"],
        lambda img: test_transforms(image=np.array(img))["image"],
    )

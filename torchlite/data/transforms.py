"""
Common data transformations
"""

from typing import Tuple, List, Union 
import numpy as np 
from PIL import Image 

class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img 
    
class ToTensor:
    """Convert PIL Image or numpy array to tensor."""

    def __call__(self, pic):
        if isinstance(pic, Image.Image):
            img = np.array(pic)

            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]

            img = img.transpose((2, 0, 1))

            return img.astype(np.float32) / 255.0
        
        elif isinstance(pic, np.ndarray):
            return pic.astype(np.float32)
        
        else:
            raise TypeError(f"Unsupported type: {type(pic)}")
        
class Normalize: 
    """Normalize a tensor with mean and standard deviation."""
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean)[:, np.newaxis, np.newaxis]
        self.std = np.array(std)[:, np.newaxis, np.newaxis]

    def __cal__(self, tensor):
        return (tensor - self.mean) / self.std 
    
class Resize:
    """Resize the input image to the given size."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size 

    def __call__(self, img):
        if isinstance(img, Image.Image):
            return img.resize(self.size, Image.BILINEAR)
        else:
            from scipy import ndimage
            factors = [s / o for s, o in zip(self.size, img.shape[:2])]
            return ndimage.zoom(img, factors + [1] * (len(img.shape) -2 ))
        
class RandomCrop:
    """Randomy crop the image"""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size 

    def __call__(self, img):
        if isinstance(img, Image.Image):
            w, h = img.size 
        else:
            h, w = img.shape[:2]
        
        th, tw = self.size 

        if w == tw and h == th:
            return img 
        
        x1 = np.random.randint(0, w - tw + 1)
        y1 = np.random.randint(0, h - th + 1)

        if isinstance(img, Image.Image):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        else:
            return img[y1:y1+th, x1:x1+tw]

class CenterCrop:
    """Crop the center of the image."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size 

    def __call__(self, img):
        if isinstance(img, Image.Image):
            w, h = img.size 
        else: 
            h, w = img.shape[:2]

        th, tw = self.size 
        x1 = (w - tw) // 2
        y1 = (h - th) // 2

        if isinstance(img, Image.Image):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        else:
            return img[y1:y1+th,x1:x1+tw]
        
class RandomHorizontalFlip:
    """Randomly flip the image horizontally."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            if isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return np.fliplr(img)
            
        return img 
    
class RandomVerticalFlip:
    """Randomly flip the image vertically."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            if isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                return np.flipud(img)
        return img 
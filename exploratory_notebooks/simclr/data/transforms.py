from torchvision import transforms
from simclr.config import CONFIG


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

def get_transforms(mean, std):

    normalize = transforms.Normalize(mean=mean, std=std)
    # image_size = 224
    EUROSAT_IMAGE_SIZE = CONFIG["EUROSAT_IMAGE_SIZE"]

    image_size = EUROSAT_IMAGE_SIZE[0]  # Use the defined EuroSAT image size 64

    eval_transform = transforms.Compose([
        transforms.Resize(image_size), # Use tuple for Resize
        transforms.CenterCrop(image_size), # Use tuple for CenterCrop
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    def get_color_distortion(s=1.0):
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort
        
    # Gaussian Blur Kernel size is 10% of image size, and must be odd
    k = int(0.1 * image_size) // 2 * 2 + 1
    gaussian_blur = transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))

    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),  
        transforms.RandomHorizontalFlip(p=0.5),
        get_color_distortion(1.0),                                
        transforms.RandomApply([gaussian_blur], p=0.5), 
        transforms.ToTensor(),
        normalize,
    ])
    
    return eval_transform, augment_transform
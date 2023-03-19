from torchvision import transforms, datasets

class ContrastiveLearningImageGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
    
    
def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)   #明亮程度、对比度、饱和色调度、色调偏移程度
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),              #将PIL图像裁剪成任意大小和纵横比
                                          transforms.RandomHorizontalFlip(),                        #以0.5的概率水平翻转给定的PIL图像
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.GaussianBlur(kernel_size=int(0.1 * size)),                #高斯滤波是应用于图像处理，对图像进行滤波操作（平滑操作、过滤操作，去噪操作
                                          ])
    return data_transforms

import yaml
from torchvision import datasets
from torchvision.transforms import transforms


#extend the ImageFolder class to return the indicies of samples as well
class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return index,img,target

def get_config(config_path=None):
    if config_path is not None:
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

            return config
    else:
        raise ValueError("config_path is not provided")

def get_torch_datasets(batch_size=32,config_path=None):#read the train and test files
    if config_path is not None:
        #read the config.yml file
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            
            train_dir = config['filepath_train']
            val_dir = config['filepath_test']

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = IndexedImageFolder(root=train_dir, transform=transform)
        val_dataset = IndexedImageFolder(root=val_dir, transform=transform)
        
       
        return train_dataset, val_dataset




from experiments import utils
from torch.utils.data import Dataset, DataLoader


class MoralStoryDataset(Dataset):

    def __init__(self,dataset_dir):
        moral_processor = utils.MoralStoriesProcessor()

        self._train_json =


if __name__ == '__main__':

    moral_processor = utils.MoralStoriesProcessor()


    training_data = moral_processor.get_train_examples("/Users/garbar/Downloads/moral_stories_datasets/classification/action+context/lexical_bias")

    print("OK")
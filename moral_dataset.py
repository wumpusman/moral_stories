from experiments import utils
from experiments import run_baseline_experiment
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import pytorch_lightning as pl
from enum import Enum
import torch

class ModelNames(Enum):
    GPT2 =  "gpt2"
    ROBERTA = "roberta"
    BAR =  "bart"
    T5 = "T5"
class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
class TaskTypes(Enum):
    ACTION_CLS = "action_cls"
    ACTION_CONTEXT_CLS = "action+context_cls"
    @staticmethod
    def get_associated_code(taskstr:str):
        return run_baseline_experiment.TASK_DICT[taskstr]

class UnknownDatasetType(Exception):pass
class UnknownTaskType(Exception):pass
class UnknownModelName(Exception):pass

class MoralStoryDataLoader(pl.LightningDataModule):
    def __init__(self,root_dir,modelname:ModelNames,tasktype:TaskTypes,tokenizer, amount_to_process = 100,batchsize = 8, num_workers = 1):
        super().__init__()
        self._root_dir = root_dir
        self._tokenizer = tokenizer
        self._model_name = modelname
        self._tasktype = tasktype
        self._amount_to_process = amount_to_process #sets a cap on how much data to select
        self._workers = num_workers
        self._batchsize = batchsize
        self._trainset = None
        self._testset = None
        self._valset = None

    def setup(self,stage=None):
        path = self._root_dir
        tokenizer = self._tokenizer
        model_name = self._model_name
        tasktype = self._tasktype

        the_moral_story_test = MoralStoryClassifyDataset(path, dataset_type=DatasetType.TEST, tokenizer=tokenizer,
                                                    model_name=model_name, tasktype=tasktype)
        the_moral_story_train = MoralStoryClassifyDataset(path, dataset_type=DatasetType.TRAIN, tokenizer=tokenizer,
                                                         model_name=model_name, tasktype=tasktype)
        the_moral_story_validation = MoralStoryClassifyDataset(path, dataset_type=DatasetType.VALIDATION, tokenizer=tokenizer,
                                                          model_name=model_name, tasktype=tasktype)

        the_moral_story_test.process_data(0,self._amount_to_process)
        the_moral_story_validation.process_data(0,self._amount_to_process)
        the_moral_story_train.process_data(0,self._amount_to_process)
        self._trainset = the_moral_story_train
        self._testset = the_moral_story_test
        self._valset =the_moral_story_validation
    def train_dataloader(self):
        return DataLoader(self._trainset,batch_size=self._batchsize,num_workers=self._workers,shuffle=True)

    def test_dataloader(self):
        return DataLoader(self._testset,batch_size=self._batchsize,num_workers=self._workers,shuffle=False)

    def val_dataloader(self):
        return DataLoader(self._valset,batch_size=self._batchsize,num_workers=self._workers,shuffle=False)


class MoralStoryDataset(Dataset):
    def __init__(self, dataset_dir, dataset_type:DatasetType = DatasetType.TRAIN, tokenizer=None):
        self._moral_processor = utils.MoralStoriesProcessor()
        self._moral_story_data  = []
        self._dataset_type_to_load = dataset_type #train, validation, "test
        self._read_data(dataset_dir)
        self._tokenizer = tokenizer
        self._cls_token = None
        self._sep_token = None
    def get_data(self):
        return self._moral_story_data
    def get_tokenizer(self):
        return self._tokenizer
    def get_cls_token(self):
        return self._cls_token
    def get_sep_token(self):
        return self._sep_token
    def get_moral_processor(self):
        return self._moral_processor
    def _process_tokenizer(self):
            self._cls_token = self._tokenizer._cls_token
            self._sep_token = self._tokenizer._sep_token
    def _read_data(self,dataset_dir):
        moral_processor = self.get_moral_processor()
        if self._dataset_type_to_load == DatasetType.TRAIN:
            self._moral_story_data = moral_processor.get_train_examples(dataset_dir)
        elif self._dataset_type_to_load == DatasetType.VALIDATION:
            self._moral_story_data = moral_processor.get_dev_examples(dataset_dir)
        elif self._dataset_type_to_load == DatasetType.TEST:
            self._moral_story_data = moral_processor.get_test_examples(dataset_dir)
        else: raise UnknownDatasetType ("unknown type specified for self._data_type_to_load: expected enums of: {}".format(DatasetType))

    def _process_data(self,moral_stories):
        pass

class MoralStoryClassifyDataset(MoralStoryDataset):
    def __init__(self, dataset_dir, dataset_type:DatasetType = DatasetType.TRAIN, tokenizer = None,
                 model_name:ModelNames = ModelNames.ROBERTA, tasktype:TaskTypes = TaskTypes.ACTION_CONTEXT_CLS):
        super().__init__(dataset_dir, dataset_type, tokenizer)
        self._model_name = model_name
        self._tasktype = tasktype
        self._tokenized_data = []
    def get_tasktype(self):
        assert "cls" in self._tasktype.value, "cls must appear for classification type task"
        return self._tasktype.value
    def get_model_name(self):
        return self._model_name

    def __getitem__(self, index):
        values =  self._tokenized_data[index]

        return {"input_ids":torch.tensor(values.input_ids),"attention_mask":torch.tensor(values.input_mask),
         "labels":torch.tensor(values.label_ids)}
    def __len__(self):
        return len(self._tokenized_data)

    def process_data(self,start_index =0 ,stop_index = -1):
        max_length = len(self._moral_story_data)
        if stop_index == -1:
            stop_index = max_length
        elif max_length < stop_index:
            stop_index = max_length

        self._tokenized_data = self._process_data(self._moral_story_data[start_index:stop_index])

    def _process_data(self,moral_stories):
        self._process_tokenizer()
        moral_processor = self.get_moral_processor()
        tokenizer = self.get_tokenizer()
        model_name = self.get_model_name()
        cls_token = self.get_cls_token()
        sep_token = self.get_sep_token()
        sep_token_extra = model_name in [ModelNames.ROBERTA]
        labels = moral_processor.get_labels()
        tasktype = self.get_tasktype()
        taskcode = TaskTypes.get_associated_code(tasktype)
        results = utils.convert_examples_to_features(moral_stories,labels,200,200,tokenizer = tokenizer ,
                                                     model_name=model_name,task_name=tasktype,
                                                     example_code=taskcode,cls_token= cls_token,
                                                     sep_token = sep_token, sep_token_extra= sep_token_extra)


        return results
if __name__ == '__main__':

    #moral_processor = utils.MoralStoriesProcessor()
    print(DatasetType.TRAIN.value)
    #the_dataset = MoralStoryDataset("/Users/garbar/Downloads/moral_stories_datasets/classification/action+context/lexical_bias","train")
    #training_data = moral_processor.get_train_examples("/Users/garbar/Downloads/moral_stories_datasets/classification/action+context/lexical_bias")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    path = "/Users/garbar/Downloads/moral_stories_datasets/classification/action+context/lexical_bias"
    datatype = DatasetType.TEST
    modeltype = ModelNames.ROBERTA
    tasktype = TaskTypes.ACTION_CONTEXT_CLS
    the_moral_story = MoralStoryClassifyDataset(path,dataset_type = DatasetType.TEST,tokenizer=tokenizer,model_name=modeltype,tasktype=tasktype)

    temp = the_moral_story.process_data(0, 10)
    print("OK")
    wat = DataLoader(the_moral_story,batch_size=2)
    result = next(iter(wat))
    print(result["input_ids"].shape)
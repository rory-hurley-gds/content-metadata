import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = ''
MODEL_PATH = "model.bin"
TRAINING_FILE = "../data/external/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) #can also use tokeizers library from hugggingface
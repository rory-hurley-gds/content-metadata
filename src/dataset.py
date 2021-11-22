import config
import torch

class EntityDataset:
    def __init__(self, texts, pos, tags):
        self.texts = texts
        self.pos = pos
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]

        ids = []
        target_pos = []
        target_tag = []

        #for each index and sentence, s, in texts
        for i, s in enumerate(text):
            
            inputs = config.TOKENIZER.encode(s, add_special_tokens=False) #encode the sentence using tokeniser in config to a list of ids(?)

            
            input_len = len(inputs) #input length is length of tokenised sentence
            ids.extend(inputs) #add tokens to ids list
            target_pos.extend([pos[i]] * input_len) #make list of pos, add to target_pos list
            target_tag.extend([tags[i]] * input_len) #make list of tags, add to target_pos list

            ids = ids[:config.MAX_LEN - 2] #take all ids, except the last two (for the first and last special tokens)
            target_pos = target_pos[:config.MAX_LEN - 2] #take all target_pos, except the last two (for the first and last special tokens)
            target_tag = target_tag[:config.MAX_LEN - 2] #take all target_tag, except the last two (for the first and last special tokens)

            ids = [101] + ids + [102] #add special token for start and end of sentences
            target_pos = [0] + target_pos + [0] #add a 0 pos to the start and end of sentences
            target_tag = [0] + target_tag + [0] #add a 0 tag to the start and end of sentences
            
            mask = [1] * len(ids) #activate an mask for every token
            token_type_ids = [0] * len(ids) #0 token type for every token in the list

            padding_len = config.MAX_LEN - len(ids) #padding is length of max layer, minus the length of the array

            ids = ids + ([0] * padding_len) #fill rest of id's with 0's for padding
            mask = mask + ([0] * padding_len) #fill rest of attention mask with 0's for padding
            token_type_ids = token_type_ids + ([0] * padding_len) #fill rest of token_type's with 0's for padding
            target_pos = target_pos + ([0] * padding_len) #fill rest of target_pos's with 0's for padding
            target_tag = target_tag + ([0] * padding_len) #fill rest of target_tag's with 0's for padding

            # return a dictionary of tensors for input into the model
            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "target_pos": torch.tensor(target_pos, dtype=torch.long),
                "target_tag": torch.tensor(target_tag, dtype=torch.long)
            }
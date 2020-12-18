
# in this file, write and utilize some helper functions for Bert
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#"sentences" should be a list of the input sentences
def tokenize_inputs_attn_mask(sentences):

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    ### do a quick test of the tokenizer.
    print(' Original: ', sentences[0])
    # Print the sentence split into tokens.
    print('Tokenized: ', tokenizer.tokenize(sentences[0]))
    # Print the sentence mapped to token ids.
    print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))


    ### Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])


    ### Add padding. We chose the max len as 50 b/c slightly longer than longest
    #   sequence in the data (38).
    MAX_LEN = 50
    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    print('\nDone.')
    #print out an example fo a padded sentence
    print(input_ids[0])


    ### add attention masks.
    attention_masks = []
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    return input_ids, attention_masks


#create training and test datasets using sklearn library.
def tr_val_split(input_ids, labels, attn_mask, test_ratio = 0.1):
    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                random_state=2018, test_size=test_ratio)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attn_mask, labels,
                                                 random_state=2018, test_size=test_ratio)
    #convert np arrays to have integer type.
    train_labels = np.array(train_labels).astype(int)
    validation_labels = np.array(validation_labels).astype(int)
    
    #convert everything to pytorch tensors
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    
    return train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks

#create iterators that automatically send data into the RAM in batches.    
def create_data_iterators(train_inputs, validation_inputs, train_labels, validation_labels,
                        train_masks, validation_masks, batch_size):
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    return train_dataloader, validation_dataloader



import os
os.environ['HF_HOME'] = "~/.cache/"
os.environ['TRANSFORMER_CACHE'] = "~/.cache/"

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoModelForQuestionAnswering,
                          AutoModelForTokenClassification)

from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AdamW
from datasets import DatasetDict
from tqdm import tqdm
import pdb
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torch.nn.functional as F
import pickle

mnli_dataset = load_dataset("multi_nli", split="train")
squad_dataset = load_dataset("squad_v2", split="train")

mnli_dataset_test = load_dataset("multi_nli", split="validation_matched")

# ignored_Index = 
# loss_squad = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)

# del mnli_dataset["validation_mismatched"]

# Get all the unique values in the genre column(Python)
# genres = set(sub_mnli_dataset["genre"])

# Get all the unique values in the genre column(NumPy arrays)
genres = mnli_dataset.unique("genre")
print(genres)
# Create a dictionary to store the different datasets
genre_subsets = {}

# Loop through the unique genres and create a new dataset for each genre
# just 2 for display purposes
genres = genres[:2]
for genre in genres:
    genre_subsets[genre] = mnli_dataset.filter(lambda example: example['genre'] == genre)

mnli_dataset_tele_test = mnli_dataset_test.filter(lambda example: example['genre'] == "telephone")

# Collect the genre-specific datasets into Hugging Face Datasets


genre_subsets_dict = DatasetDict()
for genre, dataset in genre_subsets.items():
    genre_subsets_dict[genre] = dataset

hf_model_name_or_path = "bert-base-uncased"
# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_model_name_or_path)

nli_config = AutoConfig.from_pretrained(hf_model_name_or_path, num_labels=3, finetuning_task="mnli")
# load the first model, whose bert encoder we'll use as a backbone
nli_model = AutoModelForSequenceClassification.from_pretrained(hf_model_name_or_path, config=nli_config)
qa_config = AutoConfig.from_pretrained(hf_model_name_or_path)
qa_model = AutoModelForQuestionAnswering.from_pretrained(hf_model_name_or_path, config=qa_config)


model_dict = {
    "nli": nli_model,
    "qa": qa_model
}

# copy the backbone encoder from the first model
for model in model_dict.values():
    model.bert = nli_model.bert
    # and move to cuda
    model.to("cuda")

    
sentence1_key = "premise"
sentence2_key = "hypothesis"
padding = "max_length"
max_seq_length = 128

def preprocess_function(examples):
    # Tokenize the texts
    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True)

    if "label" in examples:
        result["labels"] = examples["label"]
    return result

gov_examples = genre_subsets_dict['government'].map(
    preprocess_function,
    batched=True,
    remove_columns = genre_subsets_dict['government'].column_names,
    desc="Running tokenizer on dataset",
)

tele_examples = genre_subsets_dict['telephone'].map(
    preprocess_function,
    batched=True,
    remove_columns = genre_subsets_dict['telephone'].column_names,
    desc="Running tokenizer on dataset",
)

tele_examples_test = mnli_dataset_tele_test.map(
    preprocess_function,
    batched=True,
    remove_columns = mnli_dataset_tele_test.column_names,
    desc="Running tokenizer on dataset",
)

print("tele examples len ", len(tele_examples))  # 83348
# print("tele examples ", tele_examples)
# tele_examples = tele_examples[:8334]           
# print("tele examples len ", len(tele_examples))  

dataset_size = len(tele_examples)
indices = list(range(dataset_size))
split = int(0.1 * dataset_size)  # 10% of the dataset

# Randomly shuffle the indices
# torch.random.shuffle(indices)

# Split indices into two parts: one for the subset and one for the remaining data
subset_indices = indices[:split]
# remaining_indices = indices[split:]

# Create a Subset object for the subset
tele_subset = Subset(tele_examples, subset_indices)

tele_examples = tele_subset
print("subset tele examples len ", len(tele_examples))

gov_dataloader = DataLoader(gov_examples, collate_fn=default_data_collator, batch_size=128)
tele_dataloader = DataLoader(tele_examples, collate_fn=default_data_collator, batch_size=128)
tele_dataloader_test = DataLoader(tele_examples_test, collate_fn=default_data_collator, batch_size=200)

print("tele dataloader length ", len(tele_dataloader))


column_names = squad_dataset.column_names
pad_on_right = True
max_seq_length = 384
doc_stride = 128
pad_to_max_length  = True

question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]
answer_column_name = "answers" if "answers" in column_names else column_names[2]




# Training preprocessing
def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max_length else False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

squad_dataset = squad_dataset.map(prepare_train_features, batched=True, remove_columns=column_names)

squad_dataloader = DataLoader(squad_dataset, collate_fn=default_data_collator, batch_size=128)




optimizers = {
    "nli": AdamW(nli_model.parameters(), lr=2e-5), # large lr to see update in single step
    "qa": AdamW(qa_model.parameters(), lr=2e-5)
}
def accuracy(pred, y):
  max_preds = pred.argmax(dim = 1, keepdim = True)
#   pdb.set_trace()
  correct = (max_preds.squeeze(1)==y).float()
  return correct.sum() / len(y)


def squad_train(epochs, model, optimizer, dataloader):

    for epoch in range(epochs):
        count = 0
        epoch_acc = []
        loss_list = []
        print(f"Epoch:{epoch}")
        
        for batch in tqdm(dataloader): 
            optimizer.zero_grad()
            batch = {k: v.to("cuda") for k, v in batch.items()}
            # pdb.set_trace()
            # label = batch["labels"]\
            start_pos = batch['start_positions']
            end_pos = batch['end_positions']
            model.train()
            outputs = model(**batch)
            # pdb.set_trace()
            # logits = outputs.logits
            loss = outputs.loss
            # start_loss = loss_squad()
            loss.backward()
            optimizer.step()
            # with torch.no_grad():
            #         count+=len(label)
            #         acc = accuracy(logits, label)
            #         epoch_acc.append(acc.item())
            loss_list.append(loss.item())
        print(f"Total Number of Examples:{count}")
        print(f"Loss={sum(loss_list)/len(loss_list)}")
        # print(f"Accuracy={sum(epoch_acc)/len(epoch_acc)}")

        # save the model

        # model_save_name = "./models/squad_epoch" + str(epoch) + ".pth" 
        # torch.save({
        #      'epoch': epochs,
        #      'model_state_dict': model.state_dict(),
        #     }, model_save_name)


model = model_dict["nli"]
checkpoint = torch.load("./models/squad_epoch4.pth")
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# print(model)
# model.qa_outputs = torch.nn.Linear(768, 3)
# print(model)

# optimizer = optimizers["qa"]
# epochs = 5
# dataloader = squad_dataloader

# train(epochs, model, optimizer, dataloader)

# target_acc = evaluate(model, tele_dataloader_test)
# print(f"Accuracy on target domain:{target_acc}")

# torch.save({
#              'epoch': epochs,
#              'model_state_dict': nli_model.state_dict(),
#             }, "./models/squad_each_epoch.pth")



def evaluate(model, dataloader):
#   epoch_loss = 0
  epoch_acc = []
  model.eval()
  count = 0
  with torch.no_grad():
    for tele_batch in tqdm(dataloader):
        tele_batch = {k: v.to("cuda") for k, v in tele_batch.items()}
        labels = tele_batch["labels"]
        count+=len(labels)
        # pdb.set_trace()
        # tele_batch = tele_batch.cuda()
        predictions = model(**tele_batch)
        acc = accuracy(predictions.logits, labels)
        epoch_acc.append(acc.item())
  return sum(epoch_acc) / len(epoch_acc)


# target_acc = evaluate(model, tele_dataloader_test)
# print(f"Accuracy on target domain:{target_acc}")


#### MNLI dataset
leep_score_iter = []
acc_iter = []

def LEEP(pseudo_source_label: np.ndarray, target_label: np.ndarray):
    """
    :param pseudo_source_label: shape [N, C_s]
    :param target_label: shape [N], elements in [0, C_t)
    :return: leep score
    """
    N, C_s = pseudo_source_label.shape
    target_label = target_label.reshape(-1)
    C_t = int(np.max(target_label) + 1)   # the number of target classes
    print(C_t)
    normalized_prob = pseudo_source_label / float(N)  # sum(normalized_prob) = 1
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)
    for i in range(C_t):
        this_class = normalized_prob[target_label == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row
    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)
    empirical_prediction = pseudo_source_label @ p_target_given_source
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, target_label)])
    leep_score = np.mean(np.log(empirical_prob))
    return leep_score


def calculate_leep(model, dataloader):
#   epoch_loss = 0
  model.eval()
  count = 0
  leep_list = []
  with torch.no_grad():
    for tele_batch in tqdm(dataloader):
        tele_batch = {k: v.to("cuda") for k, v in tele_batch.items()}
        labels = tele_batch["labels"]
        count+=len(labels)
        predictions = model(**tele_batch)
        leep_score = LEEP(F.softmax(predictions.logits, dim=-1).cpu().numpy(), labels.cpu().numpy())
        leep_list.append(leep_score)
  return sum(leep_list) / len(leep_list)

def mnli_train(epochs, model, optimizer, dataloader, dataloader_test):
    for epoch in range(epochs):
        count = 0
        epoch_acc = []
        loss_list = []
        print(f"Epoch:{epoch}")
        for j,batch in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            batch = {k: v.to("cuda") for k, v in batch.items()}
            label = batch["labels"]
            model.train()
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # if j % 100 == 0:
            print(f"Epoch:{epoch} Iteration: {j}")
            with torch.no_grad():
                    count+=len(label)
                    acc = accuracy(logits, label)
                    epoch_acc.append(acc.item())
                    target_acc = evaluate(model, dataloader_test)
                    leep_score = calculate_leep(model, dataloader_test)
            # loss_list.append(loss.item())
            leep_score_iter.append(leep_score)
            acc_iter.append(target_acc)
            
            loss_list.append(loss.item())
        # print(f"Total Number of Examples:{count}")
        print(f"Loss={sum(loss_list)/len(loss_list)}")
        # print(f"Accuracy={sum(epoch_acc)/len(epoch_acc)}")

        with torch.no_grad():
            target_acc = evaluate(model, dataloader_test)
            print(f"Accuracy on target domain:{target_acc}")

# pdb.set_trace();

model.qa_outputs = torch.nn.Linear(768, 3)

optimizer = optimizers["nli"]
epochs = 5
dataloader = tele_dataloader

mnli_train(epochs, model, optimizer, dataloader, tele_dataloader_test)
leep_score = calculate_leep(model, tele_dataloader_test)
print(f"LEEP SCORE: {leep_score}")

target_acc = evaluate(model, tele_dataloader_test)
print(f"Accuracy on target domain:{target_acc}")


torch.save({
             'epoch': epochs,
             'model_state_dict': model.state_dict(),
            }, "./models/squad_mnli.pth")


with open('./leep_acc_iter.pickle', 'wb') as f:
    pickle.dump((leep_score_iter, acc_iter), f)


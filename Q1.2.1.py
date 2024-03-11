import os
os.environ['HF_HOME'] = "/scratch5/sghosal/.cache/huggingface"
os.environ['TRANSFORMER_CACHE'] = "/scratch5/sghosal"
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
from leep_score import LEEP
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt



mnli_dataset = load_dataset("multi_nli", split="train")
squad_dataset = load_dataset("squad_v2", split="train")
mnli_dataset_test = load_dataset("multi_nli", split="validation_matched")


# Get all the unique values in the genre column(NumPy arrays)
genres = mnli_dataset.unique("genre")
print(genres)
genre_subsets = {}


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
# Load the model configuration
nli_config = AutoConfig.from_pretrained(hf_model_name_or_path, num_labels=3, finetuning_task="mnli")
# load the first model, whose bert encoder we'll use as a backbone
nli_model = AutoModelForSequenceClassification.from_pretrained(hf_model_name_or_path, config=nli_config)

model_dict = {
    "nli": nli_model,
}


    
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

                          
gov_dataloader = DataLoader(gov_examples, collate_fn=default_data_collator, batch_size=32)
tele_dataloader = DataLoader(tele_examples, collate_fn=default_data_collator, batch_size=32)

tele_dataloader_test = DataLoader(tele_examples_test, collate_fn=default_data_collator, batch_size=200)

optimizers = {
    "nli": AdamW(nli_model.parameters(), lr=2e-5), # large lr to see update in single step
    # "qa": AdamW(qa_model.parameters(), lr=1e-3)
}
def accuracy(pred, y):
  max_preds = pred.argmax(dim = 1, keepdim = True)
#   pdb.set_trace()
  correct = (max_preds.squeeze(1)==y).float()
  return correct.sum() / len(y)

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


leep_score_iter=[]
acc_iter = []

def train(epochs, model, optimizer, dataloader, dataloader_test):

    for epoch in range(epochs):
        count = 0
        epoch_acc = []
        loss_list = []
       
        print(f"Epoch:{epoch}")
        
        for j, batch in tqdm(enumerate(dataloader)): 
            optimizer.zero_grad()
            batch = {k: v.to("cuda") for k, v in batch.items()}
            label = batch["labels"]
            model.train()
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if j % 100 == 0:
                print(f"Epoch:{epoch} Iteration: {j}")
                with torch.no_grad():
                        # count+=len(label)
                        # acc = accuracy(logits, label)
                        # epoch_acc.append(acc.item())
                        target_acc = evaluate(model, dataloader_test)
                        leep_score = calculate_leep(model, dataloader_test)
                # loss_list.append(loss.item())
                leep_score_iter.append(leep_score)
                acc_iter.append(target_acc)


        # print(f"Total Number of Examples:{count}")
        # print(f"Loss={sum(loss_list)/len(loss_list)}")
        # print(f"Accuracy={sum(epoch_acc)/len(epoch_acc)}")
        
        # print(f"LEEP SCORE: {leep_score}")

        with torch.no_grad():
            target_acc = evaluate(model, dataloader_test)
            print(f"Accuracy on target domain:{target_acc}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, f"/scratch5/sghosal/models/mnli_gov_tele_{epoch}.pth")

model = model_dict["nli"]
checkpoint = torch.load("/scratch5/sghosal/models/mnli_only.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = optimizers["nli"]
epochs = 5
dataloader = tele_dataloader

train(epochs, model, optimizer, dataloader, tele_dataloader_test)

target_acc = evaluate(model, tele_dataloader_test)
print(f"Accuracy on target domain:{target_acc}")

leep_score = calculate_leep(model, tele_dataloader_test)
print(f"LEEP SCORE: {leep_score}")



# with open('/scratch5/sghosal/leep_acc_iter_tele_source_transfer.pickle', 'wb') as f:
#     pickle.dump((leep_score_iter, acc_iter), f)


# plt.scatter(leep_score_iter, acc_iter)
# plt.xlabel('LEEP score')
# plt.ylabel('Validation Accuracy')
# # plt.title('Plot of X vs Y')
# # plt.grid(True)
# plt.show()
# plt.savefig("leep_visualization.png")

# torch.save({

#             'epoch': epochs,
#             'model_state_dict': nli_model.state_dict(),
#             }, "./models/mnli_only.pth")

### Evaluation Code

# checkpoint = torch.load("/scratch5/sghosal/models/mnli_gov_tel_0.pth")
# model.load_state_dict(checkpoint['model_state_dict'])

# optimizer = optimizers["nli"]
# epochs = 5
# dataloader = tele_dataloader
# model = train(epochs, model, optimizer, tele_dataloader, tele_dataloader_test)















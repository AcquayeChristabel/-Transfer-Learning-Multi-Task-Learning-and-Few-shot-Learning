import os
import numpy as np
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForTokenClassification, 
                          AutoModelForSequenceClassification, TrainingArguments, Trainer, 
                          DataCollatorForTokenClassification, AdamW, get_scheduler)
from datasets import load_dataset, load_metric
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

os.environ['HF_HOME'] = "/fs/nexus-scratch/cacquaye/cache"
os.environ['TRANSFORMERS_CACHE'] = "/fs/nexus-scratch/cacquaye/cache/"
os.environ['HF_DATASETS_CACHE'] = "/fs/nexus-scratch/cacquaye/cache/datasets"

# TRANSFORMERS_CACHE_DIR = os.getenv('TRANSFORMERS_CACHE')
# DATASETS_CACHE_DIR = os.getenv('HF_DATASETS_CACHE')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
metric_ner = load_metric("seqeval") #change name
metric_nli = load_metric("accuracy")
label_all_tokens = True
batch_size = 8


def tokenize_and_align_labels_ner(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []

    # Adjusting for the actual field names in your dataset
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Use -100 to indicate padding for Transformers models
            if word_idx is None:
                label_ids.append(-100)  # Special token or padding
            elif word_idx != previous_word_idx:  # Start of a new word
                label_ids.append(label[word_idx])
            else:
                # For subsequent subtokens of a word, we use the label of the first subtoken
                # This is because only the first subtoken is used for training in BERT-like models
                label_ids.append(-100 if not label_all_tokens else label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def process_data_ner(tokenizer, dataset_name="Babelscape/wikineural", tag_type="ner_tags"):
    # Load the datasets
    datasets = load_dataset(dataset_name, split={'train': 'train_en', 'val': 'val_en', 'test': 'test_en'})

    # Define function to tokenize and align labels
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, 
                                     is_split_into_words=True, padding='max_length', 
                                     max_length=512, return_tensors='pt')

        labels = []
        for i, label in enumerate(examples[tag_type]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:  # We set the label to -100 so they are automatically ignored in the loss function.
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx] if word_idx != None else -100)
            labels.append(label_ids)

        tokenized_inputs["labels"] = torch.tensor(labels)
        return tokenized_inputs

   
    datasets = datasets.map(tokenize_and_align_labels, batched=True)

   
    datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

   
    data_collator = DataCollatorForTokenClassification(tokenizer)
    dataloaders = {
        "train": DataLoader(datasets["train"], shuffle=True, batch_size=8),
        "val": DataLoader(datasets["val"], batch_size=8),
        "test": DataLoader(datasets["test"], batch_size=8),
    }

    return dataloaders, datasets

def process_data_nli(tokenizer, dataset_name="multi_nli"):
    datasets = load_dataset(dataset_name)

    nli_dataset = {
        "train": datasets["train"],
        "val": datasets["validation_matched"]
    }

    def tokenize_dataset_nli(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True, 
                         padding='max_length', max_length=512, return_tensors='pt')

    nli_dataset = {
        key: val.map(tokenize_dataset_nli, batched=True)
        for key, val in nli_dataset.items()
    }
    for key in nli_dataset:
        nli_dataset[key].set_format("torch")

    data_collator = DataCollatorForTokenClassification(tokenizer)
    dataloaders = {
        "train": DataLoader(nli_dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator),
        "val": DataLoader(nli_dataset["val"], batch_size=8, collate_fn=data_collator),
    }

    return dataloaders, nli_dataset


def compute_metrics_nli(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_nli.compute(predictions=predictions, references=labels)

def compute_metrics_ner(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric_ner.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }



def evaluate_ner(model, dataloader, device):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    true_labels = []

    label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels', 'token_type_ids']}
        
        with torch.no_grad():
            output = model(**batch)
        
        logits = output.logits
        preds = torch.argmax(logits, dim=2).detach().cpu().numpy()
        label_ids = batch['labels'].to('cpu').numpy()

        true_labels.extend(label_ids)
        predictions.extend(preds)
    
    # Convert predictions and true labels from ID to label string, excluding -100
    true_labels_converted = [[label_list[label] for label in sentence if label != -100] for sentence in true_labels]
    predictions_converted = [[label_list[pred] for pred, label in zip(sentence, labels) if label != -100] for sentence, labels in zip(predictions, true_labels)]
    metric_ner = load_metric("seqeval")
    metric_result = metric_ner.compute(predictions=predictions_converted, references=true_labels_converted)

    return metric_result


def evaluate_nli(model, dataloader, device):
    model.to(device)
    model.eval()

    total_correct = 0
    total_count = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)  

        with torch.no_grad():
            
            outputs = model(task='nli', input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits
        predictions = torch.argmax(logits, axis=-1)
        total_correct += (predictions == labels).sum().item()
        total_count += labels.size(0)

    accuracy = total_correct / total_count
    print(f"NLI Validation Accuracy: {accuracy}")
    return accuracy


class CustomNLINERModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model_nli = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        self.model_ner = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)
   
        self.model_nli.bert.embeddings = self.model_ner.bert.embeddings
        self.model_nli.bert.encoder = self.model_ner.bert.encoder
      
    
        assert (self.model_ner.bert.embeddings == self.model_nli.bert.embeddings), 'The models should have the same embeddings' 
        assert (self.model_ner.bert.encoder == self.model_nli.bert.encoder), 'The models should have the same encoder'

    def forward(self,task, **kwargs,):
        if task == 'nli':
            return self.model_nli(
                        
                        **kwargs
                    )
        elif task == 'ner':
            return self.model_ner(
                        
                        **kwargs
                    )
        else:
            raise ValueError("Task must be 'nli' or 'ner'")


class TaskSampler():
    def __init__(self, 
                *,
                dataloader_dict: Dict[str, DataLoader],
                task_weights=None,
                max_iters=None):
        
        assert dataloader_dict is not None, "Dataloader dictionary must be provided."

        self.dataloader_dict = dataloader_dict
        self.task_names = list(dataloader_dict.keys())
        self.dataloader_iterators = self._initialize_iterators()
        self.task_weights = task_weights if task_weights is not None else self._get_uniform_weights()
        self.max_iters = max_iters if max_iters is not None else float("inf")
        
    def __len__(self):
        return self.max_iters
    
    def _get_uniform_weights(self):
        return [1/len(self.task_names) for _ in self.task_names]
    
    def _initialize_iterators(self):
        return {name:iter(dataloader) for name, dataloader in self.dataloader_dict.items()}
    
   
    def set_task_weights(self, task_weights):
        assert sum(self.task_weights) == 1, "Task weights must sum to 1."
        self.task_weights = task_weights
    
    def get_task_weights(self):
        return self.task_weights

  
    def _sample_task(self):
        return np.random.choice(self.task_names, p=self.task_weights)
    
    def _sample_batch(self, task):
        try:
            return self.dataloader_iterators[task].__next__()
        except StopIteration:
            print(f"Restarting iterator for {task}")
            self.dataloader_iterators[task] = iter(self.dataloader_dict[task])
            return self.dataloader_iterators[task].__next__()
        except KeyError as e:
            print(e)
            raise KeyError("Task not in dataset dictionary.")
    
 
    def __iter__(self):
        self.current_iter = 0
        return self
    
    def __next__(self):
        if self.current_iter >= self.max_iters:
            raise StopIteration
        else:
            self.current_iter += 1
        task = self._sample_task()
        batch = self._sample_batch(task)
        return task, batch


def train_ner(model, datasets, tokenizer, output_dir):
    pass

def train_nli(model, datasets, output_dir):
    pass

def train_multitask_fixed_weight(multitask_model, nli_dataloaders, ner_dataloaders, task_sampler, output_dir):
    pass


def main():
    print('Hi')
    
    ner_datasets, ner_dataloaders = process_data_ner(tokenizer)
    nli_datasets, nli_dataloaders = process_data_nli(tokenizer)
    
    
    multitask_model = CustomNLINERModel().to(device)

    
    # train_ner(multitask_model.model_ner, ner_datasets, tokenizer, './fine_tuned_ner_model')

    
    # train_nli(multitask_model.model_nli, nli_datasets, './fine_tuned_nli_model')
    # Fixed weights training
    num_epochs = 3
    for epoch in range(3):
        multitask_model.train()  # Ensure the model is in training mode
        # task_sampler = TaskSampler(dataloader_dict={'ner': dataloaders_ner['train'], 
        #                                             'nli': dataloaders_nli['train']},
        #                         task_weights=[0.5, 0.5],  # Uniform scheme as baseline
        #                         max_iters=10000)  # Ensure this covers your training needs

        # task_sampler = TaskSampler(dataloader_dict={'ner': dataloaders_ner['train'], 
        #                                             'nli': dataloaders_nli['train']},
        #                         task_weights=[0.7, 0.3],  # Uniform scheme as baseline
        #                         max_iters=10000)  # Ensure this covers your training needs

        task_sampler = TaskSampler(dataloader_dict={'ner': dataloaders_ner['train'], 
                                                    'nli': dataloaders_nli['train']},
                                task_weights=[0.3, 0.7],  
                                max_iters=10000) 
        for task, batch in task_sampler:
            print(task)
         
            
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
                if task == 'nli':
             
                    outputs = multitask_model(task=task, 
                                            input_ids=batch['input_ids'], 
                                            attention_mask=batch['attention_mask'], 
                                            labels=batch['label'])

                elif task == 'ner':
                    #
                    outputs = multitask_model(task=task, 
                                            input_ids=batch['input_ids'], 
                                            attention_mask=batch['attention_mask'], 
                                            labels=batch['labels'])

                else:
                    raise ValueError(f"Unknown task: {task}")

               
                optimizer.zero_grad()
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                print(f"Epoch: {epoch}, Task: {task}, Loss: {loss.item()}")

            except TypeError as e:
                
                print(f"TypeError encountered: {e}")
                continue  
    #     # torch.save(multitask_model.state_dict(), "./Alt_0.5_0.5.pth")
#     # torch.save(multitask_model.state_dict(), "./Alt_07_03.pth")
#     torch.save(multitask_model.state_dict(), "./Alt_03_07.pth")

    # DWA here
    sigma = 0.1
    sigma = 1
    sigma = 0.5
    task_weights = [0.5, 0.5]  # Starting with uniform task weights
    num_epochs = 3
    for epoch in range(3):
        multitask_model.train()  
        task_sampler = TaskSampler(dataloader_dict={'ner': dataloaders_ner['train'], 
                                                    'nli': dataloaders_nli['train']},
                                task_weights=task_weights,  # Uniform scheme as baseline
                                max_iters=10000)  
        
        for task, batch in task_sampler:
            print(task)
            loss_history = {'ner': [], 'nli': []} 
            try:
                batch.pop('tokens', None)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
                if task == 'nli':
                    
                    outputs = multitask_model(task=task, 
                                            input_ids=batch['input_ids'], 
                                            attention_mask=batch['attention_mask'], 
                                            labels=batch['label'])

                elif task == 'ner':
                    
                    outputs = multitask_model(task=task, 
                                            input_ids=batch['input_ids'], 
                                            attention_mask=batch['attention_mask'], 
                                            labels=batch['labels'])

                else:
                    raise ValueError(f"Unknown task: {task}")
                
                optimizer.zero_grad()  
                 
                loss = outputs.loss  
                loss.backward()  
                optimizer.step()  
                
                if task in loss_history:
                    loss_history[task].append(loss.item())
                else:
                    loss_history[task] = [loss.item()]

                
                if all(len(loss_history[t]) > 1 for t in loss_history):
                    exp_weights = {}
                    for t in loss_history:
                        
                        loss_ratio = loss_history[t][-1] / loss_history[t][-2]
                        
                        exp_weights[t] = np.exp(loss_ratio / sigma)

                    
                    total_weight = sum(exp_weights.values())
                    
                    task_weights = [(exp_weights['ner'] / total_weight), (exp_weights['nli'] / total_weight)]
                    print("Task weights here")
                    print(task_weights)

                    
                    task_sampler.set_task_weights([task_weights['ner'], task_weights['nli']])
                    task_sampler.task_weights = task_weights
                
                    print(f"Epoch: {epoch}, Task: {task}, Loss: {loss.item()}")
            except TypeError as e:
                
                print(f"TypeError encountered: {e}")
                continue   
    #     # torch.save(multitask_model.state_dict(), "./Sigma_0.1_0.5_0.5.pth")
#     torch.save(multitask_model.state_dict(), "./Sigma_01.pth")
#     # torch.save(multitask_model.state_dict(), "./Sigma_05.pth")
    
    ner_results = evaluate_ner(multitask_model.model_ner, ner_dataloaders['val'], device)
    print("NER Evaluation Results:", ner_results)

    
    nli_results = evaluate_nli(multitask_model.model_nli, nli_dataloaders['val'], device)
    print("NLI Evaluation Results:", nli_results)


    
    multitask_model = custom_nli_ner_model() 
    multitask_model.load_state_dict(torch.load("./Sigma_1_0.5_0.5.pth"))  
    multitask_model.to(device)
    multitask_model.eval()  

    
    ner_dataloader = get_ner_dataloader()  
    nli_dataloader = get_nli_dataloader()  
    ner_datasets, ner_dataloader = process_data_ner(tokenizer)
    nli_datasets, nli_dataloader = process_data_nli(tokenizer)
    
    # Evaluate NER
    ner_results = evaluate_ner(multitask_model, ner_dataloader, device)
    print("NER Evaluation Results:", ner_results)

    # Evaluate NLI
    nli_results = evaluate_nli(multitask_model, nli_dataloader, device)
    print("NLI Evaluation Results:", nli_results)

    # fix error:
    label_all_tokens = True
    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    datasets = {}
    datasets['train'] = load_dataset("Babelscape/wikineural", split="train_en")
    datasets['val'] = load_dataset("Babelscape/wikineural", split="val_en")
    datasets['test'] = load_dataset("Babelscape/wikineural", split="test_en")
    tag_set = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    tag_set = {v: k for k, v in tag_set.items()}
    ner_dataset_raw = datasets
    ner_dataset = {}
    for key in ner_dataset_raw:
        ner_dataset[key] = ner_dataset_raw[key].map(tokenize_and_align_labels, batched=True)
   
        ner_dataset[key].set_format("torch")
    # ner_dataset = datasets
    ner_dataloaders = {} 
    ner_dataloaders['train'] = DataLoader(ner_dataset["train"], shuffle=True, batch_size=8, num_workers=0)
    ner_dataloaders['val'] = DataLoader(ner_dataset["val"], batch_size=4, num_workers=0)

    for key in datasets:
        datasets[key] = datasets[key].map(lambda example: {'ner_tags_named': [tag_set[tag] for tag in example['ner_tags']]})
        print(key, ':', datasets[key])
    ner_dataset_raw = datasets
    ner_dataset = {}
    for key in ner_dataset_raw:
        ner_dataset[key] = ner_dataset_raw[key].map(tokenize_and_align_labels, batched=True)
        ner_dataset[key].set_format("torch")
    data_collator = DataCollatorForTokenClassification(tokenizer)
    # dataloaders_ner, datasets_ner = (process_data(tokenizer, 'ner'))
    dataloaders_nli, nli_dataset = (process_data_nli(tokenizer, 'nli'))
    task = 'nli'
    if task=='ner':
        sample = ner_dataset["train"][:3] 
        label_key = 'labels'
    elif task=='nli':
        sample = nli_dataset["train"][:3] 
    print(sample.keys())
    labels_def = {'ner':'labels', 'nli':'label'}
    # out = multitask_model(task=task,
    #              input_ids=sample['input_ids'].cuda(), 
    #              attention_mask=sample['attention_mask'].cuda(), 
    #              labels=sample[labels_def[task]].cuda()
    #              )

    # for key in out:
    #     print(key, ":", out[key], out[key].shape)
    dataloaders_ner = ner_dataloaders
    # print(out.keys())
    print("NLI One")
    print(dataloaders_nli['train'])
    print("=================================")
    print("NER One")
    print(ner_dataloaders['train'])
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your multitask model
    print("Final Unifrom Sampling")
    multitask_model = custom_nli_ner_model()
    multitask_model.load_state_dict(torch.load("/fs/nexus-scratch/cacquaye/project/828A/HW1/Sigma_1_0.5_0.5.pth")) 
    # multitask_model.load_state_dict(torch.load("/fs/nexus-scratch/cacquaye/project/828A/HW1/Alt_07_03.pth"))  
    multitask_model.to(device)
    multitask_model.eval()  # Set the model to evaluation mode
    ner_dataloader = ner_dataloaders# 
    nli_dataloader = dataloaders_nli 

    ner_results = evaluate_ner(multitask_model, ner_dataloader['val'], device, label_list)
    print("NER Evaluation Results:", ner_results)

    # Evaluate NLI
    nli_results = evaluate_nli(multitask_model, nli_dataloader['val'], device)
    print("NLI Evaluation Results:", nli_results)   

if __name__ == "__main__":
    main()

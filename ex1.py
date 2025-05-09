from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import wandb
import os

# Initialize Weights & Biases
wandb.init(project="anlp-ex1-nevo", name="bert-base-uncased-mrpc")
CHECKPOINT_DIR = './results'
class Hyperparam:
    model_name = 'bert-base-uncased'
    epochs = 1# max 5
    learning_rate = 3e-5
    batch_size = 16


def load_training_data():
    dataset = load_dataset('glue', 'mrpc')
    tokenizer = AutoTokenizer.from_pretrained(Hyperparam.model_name)

    def preprocess_function(examples):
        return tokenizer(
            examples['sentence1'], 
            examples['sentence2'], 
            truncation=True, 
            padding='longest',  # Dynamic padding
            max_length=512  # Maximal size for Bert
        )
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Prepare the dataset for torch
    # After tokenization we use tokenized inputs (input_ids, attention_mask) and labels
    tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])
    # Trainer class expects the label column to be named 'labels'
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']
    return train_dataset, eval_dataset, tokenizer

def load_test_data():
    dataset = load_dataset('glue', 'mrpc')
    tokenizer = AutoTokenizer.from_pretrained(Hyperparam.model_name)

    def preprocess_function(examples):
        return tokenizer(
            examples['sentence1'], 
            examples['sentence2'], 
            truncation=True, 
            padding=False,  # Don't pad for prediction
            max_length=512  # Maximal size for Bert
        )
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Prepare the dataset for torch
    # After tokenization we use tokenized inputs (input_ids, attention_mask) and labels
    tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])
    # Trainer class expects the label column to be named 'labels'
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')

    test_dataset = tokenized_datasets['test']
    return test_dataset, tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = (predictions == torch.tensor(labels)).float().mean().item()
    return {'accuracy': accuracy}

def train_model():
    print("Start fine-tuning...")
    train_dataset, eval_dataset, tokenizer = load_training_data()
    # Load the model for binary classification of paraphrases
    model = AutoModelForSequenceClassification.from_pretrained(Hyperparam.model_name, num_labels=2)
    print("Dataset and model loaded successfully!")

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        learning_rate=Hyperparam.learning_rate,
        per_device_train_batch_size=Hyperparam.batch_size,
        per_device_eval_batch_size=Hyperparam.batch_size,
        num_train_epochs=Hyperparam.epochs,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=Hyperparam.epochs,# Save all checkpoints
        save_strategy='epoch',
        eval_strategy='epoch',
        report_to="wandb"  # Log to Weights & Biases
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()
    print("Model fine-tuned successfully!")

def test_model():
    # Predict using each checkpoint
    test_dataset, tokenizer = load_test_data()

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_eval_batch_size=Hyperparam.batch_size,
        report_to="wandb"  # Log to Weights & Biases
    )
    
    for checkpoint in sorted(os.listdir(CHECKPOINT_DIR)):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint)
        if os.path.isdir(checkpoint_path) and 'checkpoint' in checkpoint:
            print(f"Loading model from checkpoint: {checkpoint_path}")
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

            trainer = Trainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )
            predictions = trainer.predict(test_dataset)
            print(f"Test Results for {checkpoint}: {predictions.metrics}")

def main():
    # Log run hyperparams with Weights & Biases
    wandb.config.learning_rate = Hyperparam.learning_rate
    wandb.config.batch_size = Hyperparam.batch_size
    wandb.config.epochs = Hyperparam.epochs

    train_model()
    test_model()

if __name__ == '__main__':
    main()
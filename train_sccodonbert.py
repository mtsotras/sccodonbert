

import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import Subset
from transformers import BertConfig, BertTokenizer, BertForPreTraining
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.modeling_bert import BertPreTrainingHeads
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
dir_path = '/scratch/mmt515'

"""### Define the Custom Model Class with Regression Head"""
import torch

checkpoint_path = "/scratch/mmt515/local_CodonBERT_Melina/ScCodonBERT/pytorch_model.bin"

try:
    state_dict = torch.load(checkpoint_path)
    print("Checkpoint loaded successfully!")
    print("Keys in the state dict:", list(state_dict.keys())[:10])  # Print a few keys
except Exception as e:
    print(f"Error loading checkpoint: {e}")
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.modeling_bert import BertPreTrainingHeads
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class BertForPreTrainingWithRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    regression_logits: torch.FloatTensor = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    regression_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertForPreTrainingWithRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize BertModel with add_pooling_layer=True
        self.bert = BertModel(config, add_pooling_layer=True)
        # MLM and NSP heads
        self.cls = BertPreTrainingHeads(config)
        # Regression head
        self.regression_head = nn.Linear(config.hidden_size, 1)
        # Initialize regression head weights separately to avoid overwriting pre-trained weights
        self.regression_head.apply(self._init_weights)
        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,  # MLM labels
        next_sentence_label=None,  # NSP labels (not used in your case)
        expression=None,  # Regression labels
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get outputs from BertModel
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Ensure outputs is a ModelOutput
        )

        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        pooled_output = outputs.pooler_output       # (batch_size, hidden_size)

        # MLM and NSP predictions
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        # Regression prediction
        regression_logits = self.regression_head(pooled_output).squeeze(-1)  # (batch_size)

        # Initialize losses
        total_loss = None
        masked_lm_loss = None
        regression_loss = None

        # Compute MLM loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss

        # Compute regression loss if expression labels are provided
        if expression is not None:
            '''regression_loss = torch.sqrt(
                torch.mean((regression_logits - expression) ** 2) / regression_logits.numel()
                ) / torch.mean(expression).item()'''
            loss_fct = MSELoss()
            regression_loss = loss_fct(regression_logits, expression)
            total_loss = total_loss + 0.5*regression_loss if total_loss is not None else regression_loss

        # Return outputs using the custom ModelOutput class
        return BertForPreTrainingWithRegressionOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            regression_logits=regression_logits,
            masked_lm_loss=masked_lm_loss,
            regression_loss=regression_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

"""### Load Model and Tokenizer"""

# Paths to tokenizer and model

model_path = f"{dir_path}/local_CodonBERT_Melina/ScCodonBERT"

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load the model using the custom class
model = BertForPreTrainingWithRegression.from_pretrained(
    model_path,
    ignore_mismatched_sizes=True
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("Model and tokenizer loaded successfully.")

print(model)

"""### Load Full Data (Ground Truth DNA Sequences and Expression Values)
* Some re-naming just for fun
"""

# Load data
df = pd.read_csv(f'{dir_path}/local_CodonBERT_Melina/ScCodonBERT/Sc_cDNA_pep_TE30.csv')

print("Data loaded successfully.")

# Verify and rename columns for consistency
df.rename(columns={'te30  (log2)': 'expression', 'DNA sequence': 'dna_sequence', 'Categorical TE': 'categorical_te'}, inplace=True)

# Display the first few rows to verify
print(df.head())

"""### Pre-process data
* Create a TE token from the category of translation efficiency (i.e. TE+category = "TE10"
* Convert DNA to RNA
* Split RNA sequence into codons
* make a columns "sequence_tokens" that contains the TE and the codons
"""

# Create TE tokens
df['TE_token'] = 'TE' + df['categorical_te'].astype(str)

# Convert DNA sequences to RNA sequences
df['rna_sequence'] = df['dna_sequence'].str.upper().str.replace('T', 'U')

print("DNA sequences converted to RNA.")

def split_into_codons(sequence):
    # Split the sequence into codons of length 3
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3) if len(sequence[i:i+3]) == 3]
    return codons

# Apply the function to create a new column with codons
df['codons'] = df['rna_sequence'].apply(split_into_codons)

print("RNA sequences split into codons.")

# Combine TE tokens with codons
df['sequence_tokens'] = df.apply(lambda row: [row['TE_token']] + row['codons'], axis=1)

print("TE tokens combined with codons to form input sequences.")

print(df.head())

"""### Split Data into Train, Validation, and Test Sets"""

from sklearn.model_selection import train_test_split

# Split into training and temp datasets (temp will be split into validation and test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Split temp into validation and test datasets
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

"""### Create Hugging Face Datasets"""

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Create a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

print("Datasets created.")

"""### Define Custom Masking Function
* Use mapping of codon to amino acid
* Define custom masking function that:
    * Tokenizes without special characters (no addition of CLS, SEP, or PAD)
    * Uses first token (always TE) as CLS token (just by it being the first one, but also by setting its loss label to -100 so that it does not contribute to the MLM
    * Truncates proteins longer than 510 amino acids (BERT only takes 512 tokens)
    * Adds SEP token at the end of the sequence
    * Adds PADs if neccesary
    * Masks the TE token with prob masking_prob_TE
    * Masks codons with probability masking_prob_codons
    * Creates label tensors for training (tensors with ground thruth of masks and PAD everywhere else)
      
"""

# Define the codon to amino acid mapping (Standard Genetic Code, RNA)
codon_to_aa = {
    'UUU': 'PHE', 'UUC': 'PHE', 'UUA': 'LEU', 'UUG': 'LEU',
    'CUU': 'LEU', 'CUC': 'LEU', 'CUA': 'LEU', 'CUG': 'LEU',
    'AUU': 'ILE', 'AUC': 'ILE', 'AUA': 'ILE', 'AUG': 'MET',
    'GUU': 'VAL', 'GUC': 'VAL', 'GUA': 'VAL', 'GUG': 'VAL',
    'UCU': 'SER', 'UCC': 'SER', 'UCA': 'SER', 'UCG': 'SER',
    'CCU': 'PRO', 'CCC': 'PRO', 'CCA': 'PRO', 'CCG': 'PRO',
    'ACU': 'THR', 'ACC': 'THR', 'ACA': 'THR', 'ACG': 'THR',
    'GCU': 'ALA', 'GCC': 'ALA', 'GCA': 'ALA', 'GCG': 'ALA',
    'UAU': 'TYR', 'UAC': 'TYR', 'UAA': 'STP', 'UAG': 'STP',
    'CAU': 'HIS', 'CAC': 'HIS', 'CAA': 'GLN', 'CAG': 'GLN',
    'AAU': 'ASN', 'AAC': 'ASN', 'AAA': 'LYS', 'AAG': 'LYS',
    'GAU': 'ASP', 'GAC': 'ASP', 'GAA': 'GLU', 'GAG': 'GLU',
    'UGU': 'CYS', 'UGC': 'CYS', 'UGA': 'STP', 'UGG': 'TRP',
    'CGU': 'ARG', 'CGC': 'ARG', 'CGA': 'ARG', 'CGG': 'ARG',
    'AGU': 'SER', 'AGC': 'SER', 'AGA': 'ARG', 'AGG': 'ARG',
    'GGU': 'GLY', 'GGC': 'GLY', 'GGA': 'GLY', 'GGG': 'GLY'
}

def custom_masking(examples, masking_prob_codons, masking_prob_TE=0.0):
    sequences = examples['sequence_tokens']
    expressions = examples['expression']
    inputs = []
    max_length = 512
    max_seq_length = max_length - 1  # Reserve space for [SEP]

    for seq_tokens, expr in zip(sequences, expressions):
        input_seq = []
        labels_seq = []
        seq_tokens_upper = [token.upper() for token in seq_tokens]
        # The first token is the TE token
        TE_token = seq_tokens_upper[0]
        # Decide whether to mask the TE token
        if np.random.rand() < masking_prob_TE:
            # Mask the TE token by replacing it with 'TE0'
            input_seq.append('TE0')
        else:
            input_seq.append(TE_token)
        # **Always set the label for TE token to -100**
        labels_seq.append(-100)
        # Process the rest of the tokens (codons)
        for codon in seq_tokens_upper[1:]:
            if len(codon) != 3:
                input_seq.append(codon)
                labels_seq.append(-100)
                continue
            if np.random.rand() < masking_prob_codons:
                aa = codon_to_aa.get(codon, None)
                if aa and tokenizer.convert_tokens_to_ids(aa) != tokenizer.unk_token_id:
                    input_seq.append(aa)
                    codon_id = tokenizer.convert_tokens_to_ids(codon)
                    labels_seq.append(codon_id)
                else:
                    input_seq.append(codon)
                    labels_seq.append(-100)
            else:
                input_seq.append(codon)
                labels_seq.append(-100)
        # Truncate the input sequence and labels if necessary
        if len(input_seq) > max_seq_length:
            input_seq = input_seq[:max_seq_length]
            labels_seq = labels_seq[:max_seq_length]
        # Manually add the [SEP] token
        sep_token_id = tokenizer.sep_token_id
        input_ids = tokenizer.convert_tokens_to_ids(input_seq)
        input_ids.append(sep_token_id)
        labels_seq.append(-100)  # For [SEP]
        attention_mask = [1] * len(input_ids)
        # Pad sequences to max_length
        sequence_length = len(input_ids)
        if sequence_length < max_length:
            padding_length = max_length - sequence_length
            input_ids += [tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            labels_seq += [-100] * padding_length
        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels_tensor = torch.tensor(labels_seq)
        # Prepare the final input dictionary
        input_encoding = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_tensor,
            'expression': torch.tensor(expr, dtype=torch.float)
        }
        inputs.append(input_encoding)
    # Convert list of dictionaries to a dictionary of lists
    batch = {}
    for key in inputs[0]:
        batch[key] = torch.stack([inp[key] for inp in inputs])
    return batch

# Using the first two examples for testing
mock_data = {
    'sequence_tokens': df['sequence_tokens'].tolist()[:10],
    'expression': df['expression'].tolist()[:10]
}


mock_dataset = Dataset.from_dict(mock_data)


# Now test the custom_masking function
masked_batch = custom_masking(mock_dataset, masking_prob_codons=0.1, masking_prob_TE=0)

# Print the results
for i in range(len(mock_dataset)):
    input_ids = masked_batch['input_ids'][i]
    labels = masked_batch['labels'][i]
    expression = masked_batch['expression'][i]

    # Convert IDs to tokens
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    label_ids = labels.tolist()
    label_tokens = []
    for label_id in label_ids:
        if label_id == -100:
            label_tokens.append('[PAD]')
        else:
            label_tokens.append(tokenizer.convert_ids_to_tokens([label_id])[0])

    print(f"\nExample {i+1}:")
    print("Input Tokens:", input_tokens)
    print("Label Tokens:", label_tokens)
    print("Expression Value:", expression.item())

# Assuming df_final is your DataFrame with 'sequence_tokens' column
# Calculate the protein length (number of amino acids)
df['protein_length'] = df['sequence_tokens'].apply(lambda x: len(x) - 1)  # Subtract 1 to exclude TE token


# Number of proteins longer than 510 amino acids
num_long_proteins = (df['protein_length'] > 510).sum()

# Total number of proteins
total_proteins = len(df)

# Percentage of proteins longer than 510 amino acids
percentage_long_proteins = (num_long_proteins / total_proteins) * 100

print(f"Number of proteins longer than 510 amino acids: {num_long_proteins}")
print(f"Total number of proteins: {total_proteins}")
print(f"Percentage of proteins longer than 510 amino acids: {percentage_long_proteins:.2f}%")

"""### Define Data Collator"""

class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, masking_prob_codons=0.15, masking_prob_TE=0.0, max_length=512):
        super().__init__(tokenizer=tokenizer, padding=False)
        self.tokenizer = tokenizer
        self.masking_prob_codons = masking_prob_codons
        self.masking_prob_TE = masking_prob_TE
        self.max_length = max_length

    def set_masking_probs(self, masking_prob_codons=None, masking_prob_TE=None):
        if masking_prob_codons is not None:
            self.masking_prob_codons = masking_prob_codons
        if masking_prob_TE is not None:
            self.masking_prob_TE = masking_prob_TE

    def __call__(self, features):
        # Extract sequences and expressions from features
        sequences = [feature['sequence_tokens'] for feature in features]
        expressions = [feature['expression'] for feature in features]
        # Prepare batch for custom_masking function
        batch = {'sequence_tokens': sequences, 'expression': expressions}
        # Apply custom masking
        masked_inputs = custom_masking(batch, self.masking_prob_codons, self.masking_prob_TE)
        return masked_inputs

"""### Prepare Training Arguments"""

training_args = TrainingArguments(
    output_dir=f'{dir_path}/local_CodonBERT_Melina/ScCodonBERT/results',
    overwrite_output_dir=True,
    num_train_epochs=20,  # Adjust the number of epochs as needed
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=0.0001,
    weight_decay=0.01,
    logging_dir=f'{dir_path}/local_CodonBERT_Melina/ScCodonBERT/logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    greater_is_better=True,
    save_total_limit=2,
    remove_unused_columns=False, #Very important
    label_names=["labels", "expression"], #This was sooooo important to get the custom_metrics to work
    save_safetensors=False, #Not sure why, but not having this gives an error during training related to shared weights
    # Add any additional parameters as needed
    report_to = ["none"],
)

print("Training arguments defined.")

"""### Define Compute Metrics Function"""

print("Training arguments defined.")
import evaluate
"""### Define Compute Metrics Function"""
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Unpack logits
    prediction_scores = logits[0]  # MLM logits (batch_size, seq_len, vocab_size)
    regression_outputs = logits[2]  # Regression logits (batch_size,)
    # Unpack labels
    labels_mlm = labels[0]  # MLM labels (batch_size, seq_len)
   
    expressions = labels[1]  # Regression labels (batch_size,)

    # Convert to PyTorch tensors
    prediction_scores = torch.tensor(prediction_scores)
    labels_mlm = torch.tensor(labels_mlm)
    regression_outputs = torch.tensor(regression_outputs)
    expressions = torch.tensor(expressions)

    # Compute MLM Loss and Perplexity
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    vocab_size = prediction_scores.shape[-1]
    mlm_loss = loss_fct(
        prediction_scores.view(-1, vocab_size),  # Flatten logits
        labels_mlm.view(-1)  # Flatten labels
    )
    perplexity = torch.exp(mlm_loss).item()  # Convert loss to perplexity

    # Compute Accuracy
    mlm_preds = np.argmax(prediction_scores, axis=-1)
    labels1 = labels_mlm.reshape((-1,))
    pred1 = mlm_preds.reshape((-1,))
    idx = labels1>=0
    labels2 = labels1[idx]
    pred2 = pred1[idx]

    mlm_acc = metric.compute(predictions=pred2, references=labels2)['accuracy']
    
    # Compute Regression RMSE
    regression_outputs = regression_outputs.view(-1)  # Ensure correct shape
    expressions = expressions.view(-1)  # Ensure correct shape

    regression_outputs_np = regression_outputs.detach().cpu().numpy()
    expressions_np = expressions.detach().cpu().numpy()
    mse = mean_squared_error(expressions_np, regression_outputs_np)
    rmse = np.sqrt(mse)
    metrics = {'eval_perplexity': perplexity, 'eval_rmse': rmse, 'eval_accuracy':mlm_acc}
    # Return metrics with 'eval_' prefix for Trainer
    return metrics

"""### Initialize Trainer"""

# Initialize the custom Data Collator
data_collator = CustomDataCollator(
    tokenizer=tokenizer,
    masking_prob_codons=0.05,  # Initial masking probability for codons
    masking_prob_TE=0.0,       # Initial masking probability for TE tokens
    max_length=512
)

"""#### Custom Masking Scheduler"""

class MaskingProbCallback(TrainerCallback):
    def __init__(self, threshold=0.03, increment=0.1):
        self.previous_accuracy = 0
        self.threshold = threshold
        self.increment = increment

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Here, 'metrics' should contain eval_accuracy and other computed metrics
        current_accuracy = metrics.get('eval_accuracy', 0)
        accuracy_improvement = current_accuracy - self.previous_accuracy

        if state.epoch is not None and state.epoch > 3:
          if accuracy_improvement < self.threshold:
              new_masking_prob_codons = min(data_collator.masking_prob_codons + self.increment, 1)
              data_collator.set_masking_probs(
                  masking_prob_codons=new_masking_prob_codons,
                  masking_prob_TE=data_collator.masking_prob_TE
              )
              print(f"Low improvement ({accuracy_improvement:.2f}). Codon masking increased to {new_masking_prob_codons:.2f}")
          else:
              print(f"Accuracy improved by {accuracy_improvement:.2f}. No change in masking probabilities.")

        self.previous_accuracy = current_accuracy

# Include the callback in the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator,
    callbacks=[MaskingProbCallback],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("Trainer initialized.")
print('Initial model')
data_collator.set_masking_probs(masking_prob_codons=0.1, masking_prob_TE=0.0)
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

"""### Start Training"""

trainer.train()

"""### Evaluate model

#### Save the Fine-Tuned Model
"""

fine_tuned_model_path = f'{dir_path}/ScCodonBERT/TE_fine_tuned_models'
trainer.save_model(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)

print(f"Fine-tuned model saved to {fine_tuned_model_path}")

"""#### Test the Model on New Data"""

# Load the model using the custom class
fine_tuned_model = BertForPreTrainingWithRegression.from_pretrained(
    fine_tuned_model_path,
    ignore_mismatched_sizes=True
)

fine_tuned_model = fine_tuned_model.to(device)

trainer.model = fine_tuned_model

data_collator.set_masking_probs(masking_prob_codons=0.1, masking_prob_TE=0.0)
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

print(f"Current masking probabilities - Codons: {data_collator.masking_prob_codons}, TE: {data_collator.masking_prob_TE}")
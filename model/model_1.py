from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from google.colab import drive
drive.mount('/content/drive')

# set the device used for training
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Current device:", device)

# import the training data as a dataframe
df = pd.read_csv("/content/drive/MyDrive/stocks_processed_final.csv", encoding="utf-8")

# setting random seed, so that results are reproducible
seed = 25
np.random.seed(seed)
torch.manual_seed(seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)

# defining the input and output for the training of the model
input_col   = "headline"
target = "daily_return"

# shuffling the data will reduce potential for bias.
df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

# 80% training data, 10% validation data, 10% test data
train_df, validation_and_test = train_test_split(df, test_size=0.2, random_state=seed)
val_df, test_df = train_test_split(validation_and_test, test_size=0.5, random_state=seed)

# print the sizes for sanity check
print("Size of training data:", len(train_df))
print("Size of validation data:", len(val_df))
print("Size of test data:", len(test_df))

# getting the pre trained model
pre_trained_model = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)


# for pytorch to be able to use the data for training, we need to convert the data from
# the dataframe into pytorch tensors. We can use a custom dataset class to achieve thie.
class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, input_col, target, max_len=64):
        self.headlines   = df[input_col].astype(str).tolist()
        self.targets = df[target].astype(np.float32).values # pytorch models expect float32
        self.tokenizer = tokenizer

        # max length of 64 for the news headlines. Headlines are on median of length 55 chars.
        self.max_len = max_len

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, idx):
        headline   = self.headlines[idx]
        target = self.targets[idx]

        # run the headlines through the tokenizer to get the input ids, attention masks
        tokenized_headline = self.tokenizer(
            headline,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
            truncation=True
        )

        # we expect the tokenizer to return a dictionary with keys "input_ids", "attention_mask", and "token_type_ids"

        item = {}
        for k, v in tokenized_headline.items():
            item[k] = v.squeeze(0) # remove the batch dimension
        item["labels"] = torch.tensor(target, dtype=torch.float32)

        return item

# create custom datasets and pass them to the PyTorch dataloader
train_dataset = CustomDataset(train_df, tokenizer, input_col, target)
val_dataset   = CustomDataset(val_df,   tokenizer, input_col, target)
test_dataset  = CustomDataset(test_df,  tokenizer, input_col, target)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)


class FinBERT_Frozen_Regressor(nn.Module):
    def __init__(self, base_model_name=pre_trained_model, dropout=0.1):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)

        # freeze all the parameters of the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # note: dropout is used to avoid overfitting
        # in this regressor, we take the n-dimensional vector from FinBERT and convert it
        # to a vector of size 128. Then, it is passed to a ReLU layer and finally, to a single
        # output layer (aka, predicted daily return).
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.base_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # In BERT models, the first token is the CLS embedding, which is designed to be
        # the sentence-level representation.
        cls = outputs.last_hidden_state[:, 0, :]

        prediction = self.regressor(cls).squeeze(-1)

        # compute the regression loss
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(prediction, labels)

        return prediction, loss

# define the model and move it to the device
model = FinBERT_Frozen_Regressor()
model.to(device)

# print the model details
print("The designed model is as follows: ")
print(model)

def train_epoch(model, dataloader, optimizer, device):

    # set the model to training mode
    model.train()

    # initialize the loss to 0
    total_loss = 0.0

    i = 0
    for batch in dataloader:
        if i % 1000 == 0:
            print(f"Training batch {i}/{len(dataloader)}")
        i += 1
        # move all the batch data to GPU / CPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        labels = batch["labels"].to(device)

        # at each iteration, clear out the gradients from the last batch
        optimizer.zero_grad()

        # get the predictions and loss for the current batch
        predictions, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        loss.backward() # backpropagation to compute gradient
        optimizer.step() # perform parameter update

        total_loss += loss.item() * input_ids.size(0) # multiply by the batch size since loss is averaged

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def evaluate(model, dataloader, device):

    # set the model to evaluation mode
    model.eval()

    # initialize the loss to 0
    total_loss = 0.0

    # create list to store predictions and true labels
    all_predictions = []
    all_labels = []

    with torch.no_grad(): # disable gradient calculation
        # iterate through the batches
        for batch in dataloader:

            # move all of the batch data to the current device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # move the labels to the correct device
            labels = batch["labels"].to(device)

            # get the predictions and loss for the current batch
            prediction, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            # add the loss (multiplied by the batch size) since the loss is averaged
            total_loss += loss.item() * input_ids.size(0)

            # append the predictions and true labels
            all_predictions.append(prediction.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    # calculate the mean-squared error and mean absolute error
    mse = total_loss / len(dataloader.dataset)
    mae = mean_absolute_error(all_labels, all_predictions)

    return mse, mae

### TRAINING LOOP ###

# initialize with "infinity"
best_val_mae = float("inf")
# reasonable learning rate for BERT as we are not fine-tuning the model, but rather training the regressor
optimizer = AdamW(model.parameters(), lr=2e-4)

epoch_count = 5

# loop through the epochs
for i in range(1, epoch_count + 1):

    # get the training mean squared error by running one epoch
    train_mse = train_epoch(model, train_loader, optimizer, device)
    # get the validation mean squared error by running one epoch through the evaluate function
    val_mse, val_mae = evaluate(model, val_loader, device)

    print(f"Epoch {i}/{epoch_count} done!")
    print("Train MSE: ", train_mse, "Val MSE: ", val_mse, "Val MAE: ", val_mae)

    # if mae is lower than best seen, save the model
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), "finbert_regressor_best.pt")
        print("Saved best model.")


### TEST EVALUATION ###

# load the best saved model from disk
model.load_state_dict(torch.load("finbert_regressor_best.pt", map_location=device))
# get the test mse and mae by running the evaluate function on the test dataset
test_mse, test_mae = evaluate(model, test_loader, device)

print("\n=== Test dataset metrics ===")
print("MSE: ", test_mse, "RMSE: ", np.sqrt(test_mse), "MAE: ", test_mae)
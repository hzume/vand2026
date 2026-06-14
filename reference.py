# %%bash
# pip install -Uq pip wheel wandb
# pip install -Uq transformers["sentencepiece"]==4.44.0
# pip install -Uq peft==0.12.0 accelerate==0.33.0 bitsandbytes trl

# !nvidia-smi

# %cd /content/drive/MyDrive/05_Competition/02_atma/no17/notebook

import gc
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from jinja2 import Template

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    BitsAndBytesConfig,
)

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed
from transformers.utils import is_torch_bf16_gpu_available

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig

from trl import DataCollatorForCompletionOnlyLM

tqdm.pandas()

MODEL_NAME = "google/gemma-2-9b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get('HF_TOKEN'))
tokenizer.padding_side = "left"

df = pd.read_csv("../input/atmacup17_dataset/train.csv")
df = df.fillna("")
df = df.rename(columns={"Recommended IND":"labels"})

df

template = Template("""<bos><start_of_turn>user
Consider whether this user would recommend the product based on the reviews.
Answer Yes/No
# Review:
Age: {{age}}
Title: {{title}}
{{review_text}}

<start_of_turn>model
Answer: {{label}}""")

def preprocess_row(row:pd.Series, tokenizer:PreTrainedTokenizerBase) -> dict:
    if row["labels"] == 0:
        label_str = "No"
    else:
        label_str = "Yes"

    input_text = template.render(
        age=row["Age"],
        title=row["Title"].strip(),
        review_text=row["Review Text"].strip(),
        label=label_str,
    )
    item = tokenizer(input_text, add_special_tokens=False, truncation=False)
    return item

def preprocess_df(df:pd.DataFrame, tokenizer:PreTrainedTokenizerBase) -> pd.DataFrame:
    items = []
    for _, row in df.iterrows():
        items.append(preprocess_row(row, tokenizer))

    df = pd.concat([
        df,
        pd.DataFrame(items)
    ], axis=1)
    return df

df = preprocess_df(df, tokenizer)

class AtmaDataset(Dataset):
    def __init__(
        self,
        df:pd.DataFrame,
    ):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> dict:
        row = self.df.iloc[index]

        inputs = {
            "input_ids": row["input_ids"],
        }

        return inputs

ds = AtmaDataset(df)
data_collator = DataCollatorForCompletionOnlyLM("Answer:", tokenizer=tokenizer)
batch = next(iter(DataLoader(ds, batch_size=4, collate_fn=data_collator)))

batch["input_ids"][0]

batch["labels"][0]

print(tokenizer.decode(batch["input_ids"][0]))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    token=os.environ.get('HF_TOKEN')
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    bias='none',
    target_modules=(
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="/content/",
    overwrite_output_dir=False,

    log_level="error",

    logging_steps=10,
    logging_strategy="steps",

    eval_strategy="steps",
    eval_steps=50,
    metric_for_best_model="loss",

    save_strategy="epoch",
    save_total_limit=1,

    num_train_epochs=1,

    optim="paged_adamw_8bit",
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    learning_rate=1e-4,
    weight_decay=0.01,

    bf16=is_torch_bf16_gpu_available(),
    fp16=not is_torch_bf16_gpu_available(),

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    group_by_length=False,
    report_to='none',
    seed = 42,
    remove_unused_columns=False,
)

cv = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(df, y=df["labels"]))

fold_idx = 0
trn_idx, val_idx = cv[fold_idx]

print(f"fold: {fold_idx}")

val_idx = val_idx[:50]

print(f"train size: {len(trn_idx)}, eval size: {len(val_idx)}")

trn_df = df.iloc[trn_idx]
val_df = df.iloc[val_idx]

trainer = Trainer(
    model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=AtmaDataset(trn_df),
    eval_dataset=AtmaDataset(val_df),
    data_collator=data_collator,
)

trainer_output = trainer.train()

del model, trainer
gc.collect()
torch.cuda.empty_cache()

# !ls /content

@torch.no_grad()
def inference(ckpt_path:str, df:pd.DataFrame) -> np.ndarray:
    dl = DataLoader(
        AtmaDataset(df),
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, ckpt_path)
    model.eval()
    preds = []

    for batch in tqdm(dl):
        _ = batch.pop("labels", None)
        batch = {k:v.to(model.device.type) for k, v in batch.items()}

        # for inference shift
        batch = {k:v[:, :-1] for k, v in batch.items()}

        output = model(**batch,)
        logits = output.logits.cpu()

        class_logits = logits[:, -1, [1307, 6287]]

        preds.append(class_logits)

    preds = torch.vstack(preds).numpy()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return preds

preds = inference("/content/checkpoint-468", df.iloc[cv[0][1]])

roc_auc_score(df.iloc[cv[0][1]]["labels"].values,  softmax(preds, axis=-1)[:, -1])

test_df = pd.read_csv("../input/atmacup17_dataset/test.csv")
test_df = test_df.fillna("")
test_df["labels"] = 0 # dummy
test_df = preprocess_df(test_df, tokenizer)

test_preds = inference("/content/checkpoint-468", test_df)

sub_df = pd.read_csv("../input/atmacup17_dataset/atmaCup17__sample_submission.csv")
sub_df["target"] = softmax(test_preds, axis=-1)[:, -1]

sub_df

sub_df.to_csv("/content/submission.csv", index=False)


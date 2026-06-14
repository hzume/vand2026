import random
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
import re
import numpy as np
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd


DATA_DIR = Path('input/')
MODEL_NAME = 'facebook/nllb-200-distilled-600M'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_INPUT_LENGTH  = 256    # Maximum tokens for input question
MAX_OUTPUT_LENGTH = 512    # Maximum tokens for generated answer
BATCH_SIZE_LLM    = 8      # Reduce to 4 if you get out-of-memory errors
NUM_BEAMS         = 4      # Beam search width — higher = better quality, slower
ID_COL           = 'ID'
TEST_ID_COL      = 'ID'
QUESTION_COL     = 'input'
TEST_QUESTION_COL= 'input'
ANSWER_COL       = 'output'
LANG_COL         = 'subset'
TEST_LANG_COL    = 'subset'


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TRAIN_PATH      = DATA_DIR / 'Train.csv'
TEST_PATH       = DATA_DIR / 'Test.csv'
VAL_PATH        = DATA_DIR / 'Val.csv'
SAMPLE_SUB_PATH = DATA_DIR / 'SampleSubmission.csv'

train             = pd.read_csv(TRAIN_PATH)
test              = pd.read_csv(TEST_PATH)
val               = pd.read_csv(VAL_PATH)
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# ── Load the model and tokeniser ───────────────────────────────────────────────
print(f'Loading {MODEL_NAME}...')
print('This may take a few minutes on first run (downloading model weights).')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_llm = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    # Always load in float32 so gradient computation stays in float32.
    # fp16/bf16 mixed precision is handled by the Trainer via grad scaler,
    # not by storing the model weights in float16 directly.
    torch_dtype = torch.float32,
)
model_llm = model_llm.to(DEVICE)
model_llm.eval()

try:

    class WhitespaceTokenizer:
        """Whitespace tokeniser — language-agnostic and safe for African scripts."""
        def tokenize(self, text):
            if text is None:
                return []
            return str(text).strip().split()

    def compute_rouge(predictions, references):
        """
        Compute mean ROUGE-1 and ROUGE-L F1 scores.

        Parameters
        ----------
        predictions : list[str]
        references  : list[str]

        Returns
        -------
        dict with rouge1_f1 and rougeL_f1
        """
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rougeL'],
            tokenizer    = WhitespaceTokenizer(),
            use_stemmer  = False,
        )
        r1_scores, rl_scores = [], []

        for pred, ref in zip(predictions, references):
            score = scorer.score(str(ref), str(pred))
            r1_scores.append(score['rouge1'].fmeasure)
            rl_scores.append(score['rougeL'].fmeasure)

        return {
            'rouge1_f1': float(np.mean(r1_scores)) if r1_scores else 0.0,
            'rougeL_f1': float(np.mean(rl_scores)) if rl_scores else 0.0,
        }

    def compute_rouge_by_language(predictions, references, languages):
        """Compute ROUGE scores broken down by language."""
        results = {}
        lang_arr = np.array(languages)

        for lang in np.unique(lang_arr):
            mask    = lang_arr == lang
            preds_l = [p for p, m in zip(predictions, mask) if m]
            refs_l  = [r for r, m in zip(references,  mask) if m]
            results[lang] = compute_rouge(preds_l, refs_l)

        return pd.DataFrame(results).T

    print('✅ ROUGE scorer loaded')

except ImportError:
    print('⚠️  rouge-score not installed. Run: pip install rouge-score')
    compute_rouge = None

def make_submission(ids, predictions, output_path):
    """
    Build and save a valid Zindi submission file.

    Parameters
    ----------
    ids         : array-like of row IDs
    predictions : list[str] of generated answers
    output_path : str or Path
    """
    # Belt-and-suspenders: strip any residual sentinel tokens before saving.
    clean_preds = [re.sub(r'<extra_id_\d+>', '', str(p)).strip() for p in predictions]

    sub = pd.DataFrame()
    sub['ID']         = ids
    sub['TargetRLF1'] = clean_preds
    sub['TargetR1F1'] = clean_preds
    sub['TargetLLM']  = clean_preds

    sub = sub[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]

    # ── Submission checks ─────────────────────────────────────────────────
    required_cols = ['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']
    assert list(sub.columns) == required_cols, \
        f'Expected columns {required_cols}, got {list(sub.columns)}'
    assert len(sub) == len(test), \
        f'Row count mismatch: {len(sub)} predictions vs {len(test)} test rows'
    assert sub[['TargetRLF1', 'TargetR1F1', 'TargetLLM']].notna().all().all(), \
        'Missing values found in submission'
    assert (sub['TargetRLF1'] == sub['TargetR1F1']).all(), \
        'TargetRLF1 and TargetR1F1 differ'
    assert (sub['TargetRLF1'] == sub['TargetLLM']).all(), \
        'TargetRLF1 and TargetLLM differ'

    sub.to_csv(output_path, index=False, encoding='utf-8')
    print(f'✅ Submission saved to: {output_path}')
    print(f'   Shape : {sub.shape}')
    return sub

print(f'✅ {MODEL_NAME} loaded on {DEVICE}')
print(f'   Parameters : {sum(p.numel() for p in model_llm.parameters()) / 1e6:.0f}M')

# ── Fine-tuning configuration ──────────────────────────────────────────────
FINETUNE_OUTPUT_DIR     = './mt5-finetuned-health-qa'
FINETUNE_EPOCHS         = 3
FINETUNE_BATCH_SIZE     = 8      # Reduce to 4 if you hit OOM errors
FINETUNE_LEARNING_RATE  = 5e-5
FINETUNE_MAX_INPUT_LEN  = 256    # Must match MAX_INPUT_LENGTH used at inference
FINETUNE_MAX_TARGET_LEN = 512    # Must match MAX_OUTPUT_LENGTH used at inference
FINETUNE_VAL_SIZE       = 0.05   # 5% of training data used for validation

OUTPUT_FINETUNED = DATA_DIR / 'submission_finetuned_llm.csv'

print('Fine-tuning config:')
print(f'  Model            : {MODEL_NAME}')
print(f'  Epochs           : {FINETUNE_EPOCHS}')
print(f'  Batch size       : {FINETUNE_BATCH_SIZE}')
print(f'  Learning rate    : {FINETUNE_LEARNING_RATE}')
print(f'  Max input tokens : {FINETUNE_MAX_INPUT_LEN}')
print(f'  Max target tokens: {FINETUNE_MAX_TARGET_LEN}')
print(f'  Val split        : {FINETUNE_VAL_SIZE:.0%}')
print(f'  Output dir       : {FINETUNE_OUTPUT_DIR}')


def build_prompt(question: str, language: str = None) -> str:
    """
    Build an input prompt for the model.

    For mT5: prefix the question with a task description.
    The model learns to associate the prefix with the generation task.

    `language` may be a raw subset code (e.g. 'Amh_Eth') or a full language
    name. It is resolved through `subset_to_language_name` so the model always
    receives a human-readable language name in the prompt rather than an opaque
    code.

    Parameters
    ----------
    question : str
        The health question to answer.
    language : str, optional
        Subset code (e.g. 'Amh_Eth') or full language name. Resolved to a
        human-readable name before being inserted into the prompt.

    Returns
    -------
    str
    """
    # if language:
    #     lang_name = subset_to_language_name(language)
    #     return f'Answer this health question in {lang_name}: {question}'
    # return f'Answer this health question: {question}'
    return str(question).strip()


def generate_answers_batch(questions: list, languages: list = None,
                           batch_size: int = BATCH_SIZE_LLM) -> list:
    """
    Generate answers for a list of questions using the loaded LLM.

    Processes questions in batches to avoid out-of-memory errors.

    Parameters
    ----------
    questions : list[str]
    languages : list[str], optional
    batch_size : int

    Returns
    -------
    list[str]
    """
    if languages is None:
        languages = [None] * len(questions)

    all_answers = []
    n_batches   = (len(questions) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, len(questions))

        batch_questions = questions[start:end]
        batch_languages = languages[start:end]

        # Build prompts
        prompts = [
            build_prompt(q, l)
            for q, l in zip(batch_questions, batch_languages)
        ]

        # Tokenise
        inputs = tokenizer(
            prompts,
            return_tensors = 'pt',
            padding        = True,
            truncation     = True,
            max_length     = MAX_INPUT_LENGTH,
        ).to(DEVICE)

        # Generate
        with torch.no_grad():
            outputs = model_llm.generate(
                **inputs,
                max_new_tokens  = MAX_OUTPUT_LENGTH,
                num_beams       = NUM_BEAMS,
                early_stopping  = True,
                no_repeat_ngram_size = 3,
            )

        # Decode
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Post-process: strip mT5 sentinel tokens (<extra_id_N>) that the
        # model may emit when it has not been fine-tuned on a seq2seq task.
        # mT5 is pre-trained with a span-corruption objective that uses these
        # tokens as placeholders; a zero-shot prompt may trigger them because
        # the model has never been trained to suppress them in open generation.
        cleaned = [re.sub(r'<extra_id_\d+>', '', ans).strip() for ans in decoded]
        all_answers.extend(cleaned)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            print(f'  Batch {batch_idx + 1}/{n_batches} — {end}/{len(questions)} questions processed')

    return all_answers

print('✅ LLM generation functions defined')


# ── Build prompt-aware training dataset ───────────────────────────────────
# Critical: we use build_prompt() here so training inputs match inference
# inputs exactly. The language name (resolved from subset) is included so
# the model learns to condition its output on the target language.

def make_hf_dataset(df, question_col, answer_col, lang_col):
    """
    Convert a pandas DataFrame to a HuggingFace Dataset with prompt-formatted
    inputs and tokenised labels.

    Parameters
    ----------
    df           : pd.DataFrame
    question_col : str  — column containing the question text
    answer_col   : str  — column containing the reference answer
    lang_col     : str  — column containing the subset code (e.g. 'Amh_Eth')

    Returns
    -------
    datasets.Dataset with columns: input_ids, attention_mask, labels
    """
    records = []
    for _, row in df.iterrows():
        prompt = build_prompt(
            question = str(row[question_col]),
            language = str(row[lang_col]) if lang_col and lang_col in df.columns else None,
        )
        records.append({'prompt': prompt, 'answer': str(row[answer_col])})

    raw_ds = Dataset.from_list(records)

    def preprocess(examples):
        # Tokenise inputs (prompts)
        model_inputs = tokenizer(
            examples['prompt'],
            max_length  = FINETUNE_MAX_INPUT_LEN,
            truncation  = True,
            padding     = False,   # DataCollatorForSeq2Seq handles padding dynamically
        )
        # Tokenise targets (reference answers)
        # Use text_target= (the modern API). The older context manager
        # was removed in transformers >= 4.28 and must not be used.
        labels = tokenizer(
            text_target = examples['answer'],
            max_length  = FINETUNE_MAX_TARGET_LEN,
            truncation  = True,
            padding     = False,
        )
        # Mask padding tokens in labels so the loss ignores them.
        # Without this the model wastes capacity learning to predict pad tokens.
        label_ids = labels['input_ids']
        model_inputs['labels'] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            for seq in label_ids
        ]
        return model_inputs

    return raw_ds.map(preprocess, batched=True, remove_columns=['prompt', 'answer'])


# ── Split off a validation set from training data ──────────────────────────


train_df, val_ft_df = train_test_split(
    train,
    test_size    = FINETUNE_VAL_SIZE,
    random_state = SEED,
    stratify     = train[LANG_COL] if LANG_COL in train.columns else None,
)

print(f'Fine-tuning split — train: {len(train_df):,}  val: {len(val_ft_df):,}')

hf_train_ds = make_hf_dataset(train_df,  QUESTION_COL, ANSWER_COL, LANG_COL)
hf_val_ds   = make_hf_dataset(val_ft_df, QUESTION_COL, ANSWER_COL, LANG_COL)

print(f'HF train dataset : {hf_train_ds}')
print(f'HF val dataset   : {hf_val_ds}')

# ── Data collator — handles dynamic padding and label masking ─────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer  = tokenizer,
    model      = model_llm,
    label_pad_token_id = -100,   # already set in preprocess, belt-and-suspenders
    pad_to_multiple_of = 8,      # efficient on tensor cores
)

# ── Training arguments ─────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir                  = FINETUNE_OUTPUT_DIR,
    num_train_epochs            = FINETUNE_EPOCHS,
    per_device_train_batch_size = FINETUNE_BATCH_SIZE,
    per_device_eval_batch_size  = FINETUNE_BATCH_SIZE,
    learning_rate               = FINETUNE_LEARNING_RATE,
    predict_with_generate       = True,
    # bf16 is preferred over fp16 for seq2seq: it avoids the 'unscale FP16
    # gradients' error that occurs when model weights are in float32 but the
    # grad scaler tries to work in float16. bf16 is supported on Ampere+ GPUs
    # (A100, RTX 30xx+). Falls back to no mixed precision on older hardware.
    bf16                        = (DEVICE == 'cuda' and torch.cuda.is_bf16_supported()),
    fp16                        = (DEVICE == 'cuda' and not torch.cuda.is_bf16_supported()),
    eval_strategy               = 'epoch',              # validate after each epoch
    save_strategy               = 'epoch',
    load_best_model_at_end      = True,                 # restore best checkpoint
    metric_for_best_model       = 'eval_loss',
    logging_steps               = 100,
    generation_max_length       = FINETUNE_MAX_TARGET_LEN,
    report_to                   = 'none',               # disable W&B / MLflow
)

# ── Trainer ────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model           = model_llm,
    args            = training_args,
    train_dataset   = hf_train_ds,
    eval_dataset    = hf_val_ds,
    processing_class = tokenizer,
    data_collator   = data_collator,
)

print('Starting fine-tuning...')
print(f'  Training on {len(hf_train_ds):,} examples for {FINETUNE_EPOCHS} epoch(s)')
print(f'  Validating on {len(hf_val_ds):,} examples after each epoch')

trainer.train()

print('\n✅ Fine-tuning complete')
print(f'   Best checkpoint saved to: {FINETUNE_OUTPUT_DIR}')

# ── Regenerate test predictions with the fine-tuned model ─────────────────
# model_llm now holds the best fine-tuned checkpoint (load_best_model_at_end=True).
# We reuse generate_answers_batch() directly — it already uses model_llm
# and applies the same build_prompt() + sentinel-token cleanup.

print(f'Generating fine-tuned answers for {len(test):,} test questions...')
model_llm.eval()

test_questions_ft = test[TEST_QUESTION_COL].tolist()
test_languages_ft = test[TEST_LANG_COL].tolist() if TEST_LANG_COL else None

test_pred_finetuned = generate_answers_batch(test_questions_ft, test_languages_ft)

print(f'\n✅ Generated {len(test_pred_finetuned):,} answers')

# Preview
preview_ft = test[[TEST_ID_COL, TEST_QUESTION_COL]].copy()
preview_ft['finetuned_answer'] = test_pred_finetuned

# ── Validate on val set before saving ─────────────────────────────────────
if compute_rouge:
    val_q_ft  = val[QUESTION_COL].tolist()
    val_l_ft  = val[LANG_COL].tolist() if LANG_COL else None
    val_ref_ft = val[ANSWER_COL].tolist()

    val_pred_ft = generate_answers_batch(val_q_ft, val_l_ft)
    metrics_ft  = compute_rouge(val_pred_ft, val_ref_ft)

    print('\n📊 Fine-tuned LLM — Validation ROUGE Scores')
    print(f'   ROUGE-1 F1 : {metrics_ft["rouge1_f1"]:.4f}')
    print(f'   ROUGE-L F1 : {metrics_ft["rougeL_f1"]:.4f}')

    if LANG_COL and LANG_COL in val.columns:
        print('\n📊 ROUGE scores by language (fine-tuned model):')
        lang_metrics_ft = compute_rouge_by_language(
            val_pred_ft, val_ref_ft, val[LANG_COL].tolist()
        )

# ── Save fine-tuned submission ─────────────────────────────────────────────
print('\nSaving fine-tuned submission...')
sub_finetuned = make_submission(
    ids         = test[TEST_ID_COL].values,
    predictions = test_pred_finetuned,
    output_path = OUTPUT_FINETUNED,
)

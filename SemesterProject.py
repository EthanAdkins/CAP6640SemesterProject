"""
Resources:
Dataset/Loading:    https://huggingface.co/datasets/groloch/stable_diffusion_prompts_instruct?row=83
For Training T5:    https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
                    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb
                    https://huggingface.co/docs/transformers/model_doc/t5
                    https://huggingface.co/docs/evaluate/transformers_integrations
                    MOST USED: https://huggingface.co/docs/evaluate/transformers_integrations#seq2seqtrainer 
                    https://huggingface.co/google/flan-t5-base 
"""

from huggingface_hub import hf_hub_download
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import evaluate
from datasets import load_dataset, Dataset
import nltk
import numpy as np
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer



if __name__ == '__main__':
    runNum = 1
    modelName = "google/flan-t5-base"
    dataset = load_dataset("groloch/stable_diffusion_prompts_instruct")

    # #Debugging Dataset:
    # dataset = {
    #     "x" : ["test", "test2", "test3"],
    #     "y" : ["testtesttest", "TestTest22222", "test3Test3"]
    # }
    # dataset = pd.DataFrame(dataset)
    # dataset = Dataset.from_pandas(dataset)
    # dataset = dataset.train_test_split(test_size=0.1)

    print(dataset["train"][0])
    task_prefix = "translate basic prompt to detailed longer prompt: "

    tokenizer = T5Tokenizer.from_pretrained(modelName) # Originally used google-t5/t5-base

    def preprocess(unprocessed):
        inputs = [task_prefix + sequence for sequence in unprocessed["x"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(text_target=unprocessed["y"], max_length=1600, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess, batched=True)

    nltk.download("punkt_tab", quiet=True) # Had an issue with punkt as it is deprecated
    meteor = evaluate.load("meteor")        # See https://huggingface.co/spaces/evaluate-metric/meteor
    bertscore = evaluate.load("bertscore")  # See https://huggingface.co/spaces/evaluate-metric/bertscore
    def compute_metrics(predictions):
        preds, labels = predictions

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        meteorResult = meteor.compute(predictions=decoded_preds, references=decoded_labels)
        bertscoreResult = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        # The Seq2SeqTrainingArguments expects a dictionary of metrics/results, and bertscore returns a dictionary with a list of scores per key so we must average
        result = {
            "meteor": meteorResult["meteor"],
            "bertscore_precision": np.mean(bertscoreResult["precision"]),
            "bertscore_recall": np.mean(bertscoreResult["recall"]),
            "bertscore_f1": np.mean(bertscoreResult["f1"]),
        }
        return result

    model = T5ForConditionalGeneration.from_pretrained(modelName)   # Originally used google-t5/t5-base
    model.gradient_checkpointing_enable()                           # Improves gpu memory usage
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./results/Run' + str(runNum),
        eval_strategy='epoch',
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=5,
        num_train_epochs=5,
        fp16=False,                                 # Originally I had fp16=True but this caused issue with the flan model of nan/zero losses
        predict_with_generate=True,
        save_strategy="best",
        metric_for_best_model="loss",
        gradient_accumulation_steps=8               # Simulates larger batch sizes
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    log_history = trainer.state.log_history

    log_df = pd.DataFrame(log_history)

    log_df.to_csv("./results/Run"+str(runNum)+"/log_history.csv", index=False)


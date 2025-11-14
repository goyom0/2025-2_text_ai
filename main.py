
### main

import torch
import pandas as pd
import os
import random
import pickle
import json
import time
from tqdm import tqdm
import re
import numpy as np
import argparse
import glob
import multiprocessing as mp

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import Counter
from itertools import combinations

import sys
sys.path.append('/2025-2_text_ai')
from make_answer import LLM_model


# ===============================
# Utility: Extract & Similarity
# ===============================
def clean_generated_text(item):
    """pipeline 결과 중 텍스트만 안전하게 추출"""
    if isinstance(item, dict) and "generated_text" in item:
        return item["generated_text"].strip()
    elif isinstance(item, list) and len(item) > 0:
        if isinstance(item[0], dict) and "generated_text" in item[0]:
            return item[0]["generated_text"].strip()
        return str(item[0]).strip()
    elif isinstance(item, str):
        try:
            parsed = eval(item)
            if isinstance(parsed, list) and len(parsed) > 0 and "generated_text" in parsed[0]:
                return parsed[0]["generated_text"].strip()
        except Exception:
            return item.strip()
    return str(item).strip()


def get_all_answers(records, model, n=5, batch_size=10):
    """
    records: list[str]
    n: 반복 생성 횟수
    """
    prompt = """You are a clinician preparing an admission diagnosis list.
    Focus on observable clinical conditions rather than broad syndrome labels.
    Provide ONLY as this form:
    - Primary diagnoses: main problems (≤ 2 items, ≤ 3 words each)
    - Secondary diagnoses: comorbid or contributing conditions (≤ 2 items, ≤ 3 words each)
    - Rationale: concise reasoning (1–2 sentences)
    No additional text, no markdown.
    """

    outputs = []
    for _ in range(n):
        full_prompts = [
            f"{prompt.strip()}\n\n[Clinical Record]\n{r.strip()}" for r in records
        ]
        batch_out = model.llm.pipeline(full_prompts, batch_size=batch_size)
        cleaned_batch = [clean_generated_text(x) for x in batch_out]
        outputs.append(cleaned_batch)

    # transpose → [[case1_repeat1..n], [case2_repeat1..n], ...]
    return list(map(list, zip(*outputs)))


# ===============================
# Embedding / Evaluation
# ===============================
embedder = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

def semantic_similarity(pred, true):
    emb_pred = embedder.encode(pred, normalize_embeddings=True)
    emb_true = embedder.encode(true, normalize_embeddings=True)
    return util.cos_sim(emb_pred, emb_true).item()


def evaluate_batch(records, trues, model, n_repeat=5, batch_size=10):
    batch_preds = get_all_answers(records, model, n=n_repeat, batch_size=batch_size)
    results = {}

    for i, (record_preds, true_diag) in enumerate(zip(batch_preds, trues)):
        # semantic accuracy
        acc_scores = [semantic_similarity(p, true_diag) for p in record_preds]
        semantic_accuracy = float(np.mean(acc_scores))

        # consistency
        combs = list(combinations(record_preds, 2))
        pair_sims = [semantic_similarity(a, b) for a, b in combs]
        semantic_consistency = float(np.mean(pair_sims))
        semantic_uncertainty = 1 - semantic_consistency

        results[i] = {
            "semantic_accuracy": semantic_accuracy,
            "semantic_consistency": semantic_consistency,
            "semantic_uncertainty": semantic_uncertainty,
            "predictions": record_preds,
            "true_diagnosis": true_diag,
        }
    return results


# ===============================
# Main
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    HF_TOKEN = 'hf_key'
    data = pd.read_csv("/2025-2_text_ai/data/test_data.csv", index_col=0)
    
    # model_id = 'meta-llama/Llama-3.2-3B-Instruct'
    model_id = args.model
    # meta-llama/Meta-Llama-3.3-70B-Instruct

    temperature = args.temp

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=512,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        return_full_text=False,
        add_special_tokens=False,
    )

    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

    from make_answer import LLM_model
    model = LLM_model(model_id=model_id, pipe=pipe, tokenizer=tokenizer)

    random.seed(0)
    # ran_samples = random.sample(list(data.index), 1000)
    # data = data.loc[ran_samples].sort_index(ascending=True)

    batch_size = 10
    n_repeat = 5
    res = {}
    save_dir = "/2025-2_text_ai/result"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_id.replace("/", "_") + "_result.json")

    try:
        for start in tqdm(range(0, len(data), batch_size)):
            batch_idx = data.index[start:start + batch_size]
            batch_records = [
                f"#Patient Info : {data.loc[i, 'patient_info']}\n{data.loc[i, 'HPI']}"
                for i in batch_idx
            ]
            batch_trues = [data.loc[i, 'diagnosis'] for i in batch_idx]
            batch_res = evaluate_batch(batch_records, batch_trues, model,
                                       n_repeat=n_repeat, batch_size=batch_size)
            for j, sid in enumerate(batch_idx):
                res[str(sid)] = batch_res[j]

    finally:
        try:
            with open(save_path, "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
        except Exception as fe:
            print(f"[SAVE ERROR] {fe}")


if __name__ == "__main__":
    main()

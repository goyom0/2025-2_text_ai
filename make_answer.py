
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
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import sys
sys.path.append('/2025-2_text_ai/project')

import unicodedata


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE)

LANGCHAIN_API_KEY ='langchain_key'
OPENAI_API_KEY = 'openai_key'
HF_TOKEN = 'hf_key'

os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['DEVICE'] = DEVICE


### making answers

class LLM_model:
    def __init__(self, model_id: str, pipe, tokenizer, temperature: float = 0.7, repetition_penalty: float = 1.2):
        self.model_name = model_id
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        
        if "gpt" in model_id.lower():
            self.llm_type = "gpt"
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=temperature,
                openai_api_key=OPENAI_API_KEY,
            )
        elif "llama" in model_id.lower():
            # self.model = model
            self.pipe = pipe
            self.llm_type = "llama"
            self.llm = HuggingFacePipeline(pipeline=pipe)
        else:
            raise ValueError("Unsupported model type.")

    def chain_result(self, clinical_record, prompt):
        system = """You are a clinician reviewing a patient’s hospital admission record.
            Your task is to identify the main clinical diagnoses based on the record.
            Focus on concise, high-level clinical diagnoses that summarize the presentation —
            not raw symptoms (too low-level) and not mechanistic etiologies (too deep).
            Prefer general, documentation-level diagnoses over inferred causes.
            Limit your reasoning depth to what is clearly supported by the record.
            If information is incomplete, state only what can be directly supported."""

        if self.llm_type == "gpt":
            prompt_tmpl = ChatPromptTemplate.from_messages([
                ("system", system),
                ("user", f"{prompt.strip()}\n{clinical_record.strip()}"),
            ])
            chain = (
                {"clinical_record": RunnablePassthrough()}
                | prompt_tmpl
                | self.llm
            )
            result = chain.invoke({"clinical_record": clinical_record})
            return result.content if hasattr(result, "content") else result

        elif self.llm_type == "llama":
            # system + user 메시지를 하나의 텍스트로 구성
            full_prompt = (
                f"[System]\n{system.strip()}\n\n"
                f"[User]\n{prompt.strip()}\n\n"
                f"[Clinical Record]\n{clinical_record.strip()}\n\n"
                "### Response:\n"  # 구분 마커
            )
            result = self.llm.pipeline(full_prompt)

            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                text = result[0]["generated_text"]
            else:
                text = str(result)

            # "### Response" 이후만 추출
            if "### Response" in text:
                text = text.split("### Response", 1)[-1]

            return text.strip()


    ### 첫번째 답변에서 핵심 diagnosis 뽑기 ---> format 지정 어려워 사용하지 X
    def get_diagnosis(self, model_output):

        prompt = f"""
            You are not a clinician. You are a string extraction system.
            Your task: extract at most 3 diagnosis names from the text below.

            Constraints:
            - Output only the diagnosis names that explicitly appear in the text.
            - Each must be short (≤ 5 words).
            - No explanations, no reasoning, no commentary.
            - No Markdown, headings, numbering, punctuation beyond the hyphen.
            - Do NOT write "final answer", "answer:", or anything similar.
            - Output only the list, like this:
                1. Diagnosis A
                2. Diagnosis B
                3. Diagnosis C

            If no explicit diagnosis is found, return nothing.

            Text:
            {model_output}

            ### Response:
            """

        if self.llm_type == "gpt":
            diag = self.llm.invoke(prompt)
            diag = diag.content if hasattr(diag, "content") else diag
        else:  # llama
            diag = self.llm.pipeline(prompt)
            diag = diag[0]["generated_text"].strip()

        if "### Response:" in diag:
            diag = diag.split("### Response:", 1)[-1]

        diag = diag.strip()
        diag_text = re.sub(r"```[\s\S]*?```", "", diag)   # 코드 블록 제거
        diag_text = re.sub(r"[ \t]+", " ", diag_text)
        diag_text = re.sub(r"\s*\n\s*", "\n", diag_text)
        diag_text = diag_text.strip()

        lines = [
            ln.strip()
            for ln in diag_text.split("\n")
            if re.match(r"^\s*\d+\.\s*[A-Za-z가-힣]", ln)
        ]
        diag_text = "\n".join(lines)

        print(diag_text)

        return {"reasoning": model_output, "diagnosis": diag}



# project code 정리

# 1. ID 매핑: evaluate_batch_multicandidate 함수에 ids 인자를 추가하여 실제 데이터의 ID(subject_id가 없다면 인덱스)를 사용
# 2. Rationale 추가: 프롬프트 수정 - 진단명과 함께 근거를 출력하게 하고, 결과 딕셔너리에 rationales 추가 -- 삭제
# 3. RAG On/Off: use_rag 추가, False일 경우 검색 과정을 건너뛰고 HPI만 입력받도록 함
    # Usage: python project_final.py --model meta-llama/Llama-3.2-3B-Instruct (--rag)
    # --rag 플래그 있으면 rag 실행, 없으면 baseline (rag 없음)

import pandas as pd
import numpy as np
import json
import re
import os
import random
import argparse
from tqdm import tqdm
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from dotenv import load_dotenv

# === HF 토큰 설정 ===
HF_TOKEN = input("Enter your Hugging Face token:")
login(token=HF_TOKEN)


# ===== 설정 =====
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# === 기본 설정 ===
CONFIDENCE_THRESHOLD = 0.6
DATA_PATH = "/home/work/.dahyoun/class/text_ai/project/data/test_data.csv"
# 이미 rag에 쓸 요약본 있는 경우
# DATA_PATH = "/home/work/.dahyoun/class/text_ai/project/data/rag.csv"

# === Semantic 임베딩 ===
print("Semantic 임베딩 로드 중...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# === RAG 데이터 전처리 모델 ===
print("LLM 전처리 모델 로드 중...")
llm_preprocess_pipe = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.2-3B-Instruct",
    tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    max_new_tokens=256, 
    temperature=0.1,
    do_sample=True,
    device_map="auto"
)

llm_preprocess_pipe.tokenizer.pad_token = llm_preprocess_pipe.tokenizer.eos_token
llm_preprocess_pipe.tokenizer.pad_token_id = llm_preprocess_pipe.tokenizer.eos_token_id
llm_preprocess_pipe.model.config.pad_token_id = llm_preprocess_pipe.model.config.eos_token_id


# ==========================================
# [Helper Functions] 
# ==========================================

def semantic_similarity(text1, text2):
    try:
        t1 = str(text1).lower().strip()
        t2 = str(text2).lower().strip()
        if not t1 or not t2: return 0.0
        emb1 = embedder.encode(t1, normalize_embeddings=True)
        emb2 = embedder.encode(t2, normalize_embeddings=True)
        return float(util.cos_sim(emb1, emb2))
    except:
        return 0.0

def calculate_set_similarity(list1, list2):
    """Set-to-Set Similarity (Max Pooling Mean)"""
    if not list1 or not list2: return 0.0
    
    scores_1_to_2 = []
    for t1 in list1:
        # t1이 list2의 항목 중 가장 비슷한 것과 얼마나 유사한지
        best_score = max([semantic_similarity(t1, t2) for t2 in list2]) if list2 else 0
        scores_1_to_2.append(best_score)
    
    scores_2_to_1 = []
    for t2 in list2:
        best_score = max([semantic_similarity(t2, t1) for t1 in list1]) if list1 else 0
        scores_2_to_1.append(best_score)
        
    return (np.mean(scores_1_to_2) + np.mean(scores_2_to_1)) / 2.0


def extract_info(text: str):
    text = str(text)
    m = re.search(r'Gender:\s*(.*?),\s*Race:\s*(.*?),\s*Age:\s*(\d+)', text)
    if not m: return None, None, None
    gender = m.group(1).strip()
    race   = m.group(2).strip()
    age    = int(m.group(3))
    start = (age // 10) * 10
    end   = start + 10
    return gender, race, f"{start}-{end}"

# ==========================================
# [Improved Extraction Logic] 
# ==========================================

def validate_llm_output(text):
    text = str(text).lower().strip()
    
    # 1. 전처리: 불필요한 접두어/특수문자 제거
    # "1. ", "- ", "is ", "the " 제거
    text = re.sub(r'^[\d\.\-\)\•\s]+', '', text)
    text = re.sub(r'^(is|the|a|an|possible|likely)\s+', '', text)
    text = re.sub(r'[^\w\s]', '', text) # 특수문자 제거
    
    # 2. 기본 필터 (길이 및 금지어)
    if len(text) < 3 or len(text) > 60: return None
    invalid_keywords = [
        'unknown', 'cannot', 'insufficient', 'clinician', 'doctor', 
        'diagnosis', 'diagnoses', 'none', 'n/a', 'answer',
        'rationale', 'explanation', 'evidence', 'subtype', 'reasoning',
        'history', 'symptoms', 'based on'
    ]
    if any(k in text for k in invalid_keywords): return None

    # 3. [복구됨] Medical Keyword Whitelist (허용 단어 목록)
    # 이 단어들이 포함되어야만 진단명으로 인정 (기존 로직의 핵심)
    medical_keywords = [
        # 질병 접미사
        'itis', 'osis', 'ia', 'oma', 'pathy', 'megaly', 'emia', 
        # 주요 질병/증상 키워드
        'pain', 'fracture', 'syndrome', 'disease', 'disorder', 'failure', 'injury',
        'cancer', 'tumor', 'mass', 'abscess', 'cyst', 'infarction', 'stroke',
        'pneumonia', 'sepsis', 'anemia', 'delirium', 'leukemia', 'lymphoma',
        'diabetes', 'hypertension', 'copd', 'chf', 'asthma', 'bleed', 'hemorrhage',
        'infection', 'effusion', 'embolism', 'sclerosis', 'cirrhosis', 'hepatitis',
        'calculus', 'stone', 'obstruction', 'ileus', 'hernia', 'ulcer', 'gastritis',
        'pancreatitis', 'appendicitis', 'cholecystitis', 'uti', 'renal', 'kidney',
        'liver', 'heart', 'lung', 'brain', 'abdominal', 'chest', 'acute', 'chronic'
    ]
    
    # 텍스트 안에 의학 키워드가 하나라도 있는지 확인
    if not any(keyword in text for keyword in medical_keywords):
        return None
        
    return text.strip()

def extract_diagnoses_from_text(text):
    """
    텍스트에서 진단명을 리스트로 추출 (여러 개 탐색 + 강력한 검증)
    """
    candidates = []
    text_lower = str(text).lower()

    # 1. 포맷 기반 추출 ("Diagnosis: ...")
    # 줄바꿈이나 'rationale' 나오기 전까지만 가져옴
    format_matches = re.findall(r'diagnosis:?\s*(.*?)(?:\s*-\s*rationale|\n|$)', text_lower)
    if not format_matches:
        format_matches = re.findall(r'diagnoses:?\s*(.*?)(?:\s*-\s*rationale|\n|$)', text_lower)
        
    for match in format_matches:
        # 콤마로 연결된 경우 분리 (ex: "Pneumonia, UTI")
        sub_parts = re.split(r',|;', match)
        for part in sub_parts:
            valid = validate_llm_output(part)
            if valid: candidates.append(valid)

    # 2. 리스트/불렛 기반 추출 (1. ..., - ...)
    if not candidates:
        list_matches = re.findall(r'(?:^\d+[\.\)]|^[-•\*])\s*(.*?)(?:\n|$|-)', text_lower, re.MULTILINE)
        for match in list_matches:
            sub_parts = re.split(r',|;', match)
            for part in sub_parts:
                valid = validate_llm_output(part)
                if valid: candidates.append(valid)

    # 3. 키워드 패턴 매칭
    if not candidates:
        # 3-1. 특정 패턴 (acute OO, OO fracture 등)
        patterns = [
            r'\b(acute|chronic)\s+([a-z]+)\b',
            r'\b([a-z]+)\s+(fracture|pain|syndrome|disease|failure)\b',
            r'\b(pneumonia|sepsis|chf|copd|infarction|anemia|leukemia)\b'
        ]
        for pat in patterns:
            hits = re.findall(pat, text_lower)
            for hit in hits:
                if isinstance(hit, tuple): hit = " ".join(hit)
                valid = validate_llm_output(hit)
                if valid: candidates.append(valid)
                
        # 3-2. 긴 단어(5글자 이상) 중 Medical Keyword 통과하는 것
        words = re.findall(r'\b[a-z]{5,}(?:\s+[a-z]{3,})*\b', text_lower)
        for w in words:
            valid = validate_llm_output(w)
            if valid: candidates.append(valid)

    return list(set(candidates))


def parse_diagnosis_json(text):
    text = str(text)

    # JSON 배열이 아예 없는 경우
    if "[" not in text or "]" not in text:
        return []

    try:
        # 첫 번째 [ ... ] 부분만 파싱
        start = text.index("[")
        end = text.index("]", start) + 1
        json_str = text[start:end]

        # 작은따옴표 → 큰따옴표 자동변환
        json_str = json_str.replace("'", '"')

        arr = json.loads(json_str)

        # 문자열만 남기기
        return [str(x).strip() for x in arr if len(str(x).strip()) > 1]
    except Exception as e:
        return []



# ==========================================
# [Batch Processing]
# =========================================

# 1. Dataset 클래스
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list
    def __len__(self):
        return len(self.original_list)
    def __getitem__(self, i):
        return self.original_list[i]

# 2. 프롬프트 준비 함수 (Batch 전처리)
def prepare_prompt_batch(hpi_text, summary_text, vectorizer, rag_vectors, rag_df, use_rag=True):
    # 요약본 사용
    q_text = summary_text if summary_text and str(summary_text) != "nan" else hpi_text
    bm25_candidates = []
    context = ""

    if use_rag:
        q_vec = vectorizer.transform([q_text])
        sims = cosine_similarity(q_vec, rag_vectors).flatten()
        
        top_idx_diag = np.argsort(sims)[-7:][::-1]
        bm25_candidates = [diag for sublist in rag_df.iloc[top_idx_diag]['diagnosis_list'] for diag in sublist]
        
        top_idx_full = np.argsort(sims)[-3:][::-1]
        similar_cases = rag_df.iloc[top_idx_full][['llm_hpi_summary', 'diagnosis_list']].to_dict('records')
        context = "Here is some similar cases:\n" + "\n".join([
            f"{i+1}. Patient HPI: {case['llm_hpi_summary']} → Diagnosis: {', '.join(case['diagnosis_list'])}" 
            for i, case in enumerate(similar_cases)
        ])
    else:
        context = "No similar cases provided."

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a skilled clinician. Provide up to 3 possible diagnoses from HPI.

        Return ONLY valid JSON.
        No explanations. No additional text.
        Format:["...","...","..."]
        Similar Cases:
        {context} \n\n

        Patient HPI:
        {hpi_text}

        Diagnosis:
        """
    
    meta_data = {"bm25_candidates": bm25_candidates[:3]}
    return prompt, meta_data


def process_llm_output(llm_out, meta_data, use_rag=True):
    current_candidates = []
    
    # 배치 출력은 num_return_sequences=3 이므로 리스트 형태
    for single_gen in llm_out:
        generated_text = single_gen['generated_text']
        # print("\n========== RAW LLM OUTPUT ==========")
        # print(generated_text)
        # print("====================================\n")
        extracted = parse_diagnosis_json(generated_text)
        current_candidates.extend(extracted)

    if use_rag:
        hybrid_candidates = meta_data['bm25_candidates'] + current_candidates
    else:
        hybrid_candidates = current_candidates
        
    if not hybrid_candidates:
        return ["unknown"]
        
    candidate_counts = Counter(hybrid_candidates)
    top_3 = [c for c, _ in candidate_counts.most_common(5)][:3]
    return top_3


def get_all_answers_multicandidate(records, summaries, vectorizer, rag_vectors, rag_df, pipe, n=5, use_rag=True):
    print("🔄 [Step 1] Preparing Prompts...")
    prompts = []
    meta_datas = []
    
    # 길이 맞추기
    if len(records) != len(summaries):
        min_len = min(len(records), len(summaries))
        records = records[:min_len]
        summaries = summaries[:min_len]

    # 프롬프트 준비
    for hpi, summ in tqdm(zip(records, summaries), total=len(records)):
        p, m = prepare_prompt_batch(hpi, summ, vectorizer, rag_vectors, rag_df, use_rag)
        prompts.append(p)
        meta_datas.append(m)

    # 리스트를 Dataset으로 변환
    dataset = ListDataset(prompts)

    all_preds = [[] for _ in range(len(prompts))]
    
    for i in range(n):
        print(f"🚀 [Step 2] Batch Inference Run {i+1}/{n}")
        batch_results = []
        
        # pipe에 dataset 전달
        # num_return_sequences=3이므로 결과는 [[dict, dict, dict], ...] 형태
        for out in tqdm(pipe(dataset, batch_size=32, num_return_sequences=3), total=len(dataset)):
            batch_results.append(out)
        
        for idx, (llm_outs) in enumerate(batch_results):
            top_3 = process_llm_output(llm_outs, meta_datas[idx], use_rag)
            all_preds[idx].append(top_3)
            
    return all_preds


def evaluate_batch_multicandidate(records, summaries, trues, ids, args, pipe, n_repeat=5, use_rag=True):
    vectorizer, rag_vectors, rag_df = args
    
    batch_preds = get_all_answers_multicandidate(records, summaries, vectorizer, rag_vectors, rag_df, pipe, n_repeat, use_rag)
    
    results = {}
    print("🔄 [Step 3] Calculating Metrics...")
    
    for i, (record_preds_list, true_diag, patient_id, hpi_text) in tqdm(enumerate(zip(batch_preds, trues, ids, records)), total=len(records)):
        
        if not isinstance(true_diag, list): 
            true_diag = normalize_diagnosis(true_diag)
            
        # Accuracy
        acc_scores = [calculate_set_similarity(preds, true_diag) for preds in record_preds_list]
        semantic_accuracy = float(np.mean(acc_scores))

        # Consistency
        if len(record_preds_list) > 1:
            from itertools import combinations
            run_pairs = list(combinations(record_preds_list, 2))
            pair_scores = [calculate_set_similarity(p1, p2) for p1, p2 in run_pairs]
            semantic_consistency = float(np.mean(pair_scores))
        else:
            semantic_consistency = 1.0
            
        results[patient_id] = {
            "patient_id": patient_id,
            "hpi": hpi_text[:200] + "...",
            "true_diagnosis": true_diag,
            "predictions_per_repeat": record_preds_list, 
            "semantic_accuracy": semantic_accuracy,         
            "semantic_consistency": semantic_consistency,   
            "semantic_uncertainty": 1.0 - semantic_consistency,
            "use_rag": use_rag
        }
    return results


# ==========================================
# [Main Execution]
# ==========================================

def load_main_llm(model_name):
    print("메인 LLM 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.config.pad_token_id = model.config.eos_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        return_full_text=False,
        repetition_penalty=1.3,
        add_special_tokens=False,
    )
    
    return pipe






def main():
    import os
    import argparse
    
    global embedder
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--rag", action="store_true", help="enable RAG")
    args = ap.parse_args()

    seed = 0
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

    # === 1. 데이터 로드 ===
    print("=== Data Load ===")
    # DATA_PATH = "/home/work/.dahyoun/class/text_ai/project/data/rag.csv"
    df = pd.read_csv(DATA_PATH, index_col=0)
    df = df.sort_values(by="stay_id")
    
    # ID 컬럼 통일 (stay_id가 없으면 인덱스 사용)
    if 'stay_id' not in df.columns:
        df['stay_id'] = df.index.astype(str)

    df = df.dropna(subset=['HPI','patient_info', 'diagnosis']).reset_index(drop=True)

    # df sampling
    # df = df.sample(frac=0.15, random_state=seed)
    df = df.sample(n=1000, random_state=seed)
    
    # Train/Test 분리
    rag_df = df.sample(frac=0.8, random_state=seed)
    test_df = df.drop(rag_df.index).reset_index(drop=True)

    
    # 필수 컬럼 확인 (없으면 에러 방지 위해 임시 생성)
    if 'llm_hpi_summary' not in df.columns:
        # === 검색용 텍스트 (HPI 요약 + 진단명 합치기) ====    
        print("=== Start data preprocessing ===")
        rag_df['search_text'] = rag_df['llm_hpi_summary'] + ' ' + ' ' +  rag_df['patient_info_extract'].apply(
            lambda x: ' '.join([str(v) for v in x if v]) if isinstance(x, tuple) else str(x)
        ) + ' ' + rag_df['diagnosis_list'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        test_df['query_text'] = test_df['llm_hpi_summary']
        
        print("=== Vectorizer ===")
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
        rag_vectors = vectorizer.fit_transform(rag_df['search_text'])
        test_queries = vectorizer.transform(test_df['query_text'])
        
        print(f"✅ LLM 전처리 완료: {len(rag_df)} RAG / {len(test_df)} Test")

        # 요약 없이 원본 사용할 경우
        # print("⚠️ 'llm_hpi_summary' 컬럼이 없어 HPI 원본을 대신 사용합니다.")
        # df['llm_hpi_summary'] = df['HPI']
    
    # patient_info_extract가 없으면 추출 수행
    if 'patient_info_extract' not in df.columns:
        print("ℹ️ patient_info 추출 수행 중...")
        df['patient_info_extract'] = df['patient_info'].apply(extract_info)


    # === 검색용 인덱스 생성 ===
    print("=== Start data preprocessing ===")
    # 정규화
    rag_df['diagnosis_list'] = rag_df['diagnosis'].apply(normalize_diagnosis)
    test_df['diagnosis_list'] = test_df['diagnosis'].apply(normalize_diagnosis)
    
    # 검색 텍스트 생성
    rag_df['search_text'] = rag_df['llm_hpi_summary'].astype(str) + ' ' + rag_df['patient_info_extract'].apply(
        lambda x: ' '.join([str(v) for v in x if v]) if isinstance(x, tuple) else str(x)
    ) + ' ' + rag_df['diagnosis_list'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

    print("=== Vectorizer ===")
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
    rag_vectors = vectorizer.fit_transform(rag_df['search_text'])

    print(f"✅ RAG 준비 완료: {len(rag_df)} docs")

    # === 4. 메인 LLM 로드 ===
    model_path = args.model 
        
    # print(f"메인 LLM 로드 중: {model_path}")
    llm_pipe = load_main_llm(model_path)

    # === 5. Test 시작 ===
    try:
        print("=== Start Test ===")
        test_subset = test_df 
        hpi_samples = test_subset['HPI'].tolist()
        true_samples = test_subset['diagnosis_list'].tolist()
        id_samples = test_subset['stay_id'].tolist() # stay_id 사용 통일
        
        # 요약본 리스트 추출
        test_summaries = test_subset['llm_hpi_summary'].fillna("").astype(str).tolist()

        USE_RAG_OPTION = args.rag 
        if USE_RAG_OPTION:
            print("Using RAG!")

        # 평가 함수 호출
        results = evaluate_batch_multicandidate(
            records=hpi_samples, 
            summaries=test_summaries,
            trues=true_samples, 
            ids=id_samples, 
            args=(vectorizer, rag_vectors, rag_df),
            pipe=llm_pipe, 
            n_repeat=5, 
            use_rag=USE_RAG_OPTION
        )

    finally:
        print("=" * 80)
        if 'results' in locals() and results:
            # Output preview
            first_key = list(results.keys())[0]
            res = results[first_key]
            print(f"   Patient ID: {res['patient_id']}")
            print(f"   True: {res['true_diagnosis']}")
            print(f"   Prediction: {res['predictions_per_repeat'][0]}")
            print(f"   *** Accuracy: {res['semantic_accuracy']:.3f}")
            
            M = args.model.split('/')[-1].replace("-", "_")
            rag_str = 'rag' if args.rag else 'base'
            
            savedir = "/home/work/.dahyoun/class/text_ai/project/final_result"
            os.makedirs(savedir, exist_ok=True)
            output_path = f"{savedir}/{M}_{rag_str}_results.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"💾 JSON 저장 완료: {output_path}")



if __name__ == "__main__":
    main()

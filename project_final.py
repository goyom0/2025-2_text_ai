

# project code 정리

# 1. ID 매핑: evaluate_batch_multicandidate 함수에 ids 인자를 추가하여 실제 데이터의 ID(subject_id가 없다면 인덱스)를 사용
# 2. Rationale 추가: 프롬프트 수정 - 진단명과 함께 근거를 출력하게 하고, 결과 딕셔너리에 rationales 추가 -- 삭제
# 3. RAG On/Off: hybrid_rag_llm_llm_preprocess 함수에 use_rag 파라미터 추가, False일 경우 검색 과정을 건너뛰고 HPI만 입력받도록 함
    # Usage: python project_final.py --model meta-llama/Llama-3.2-3B-Instruct (--rag)
    # --rag 플래그 있으면 rag 실행, 없으면 baseline (rag 없음)


import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import random

# === HF 토큰 설정 ===
HF_TOKEN = input("Enter your Hugging Face token:")
login(token=HF_TOKEN)

# === 기본 설정 ===
CONFIDENCE_THRESHOLD = 0.6
DATA_PATH = "/home/work/.dahyoun/class/text_ai/project/data/test_data.csv"

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

# ===== 3.  전처리 함수들 =====

# diagnosis 정규화
def normalize_diagnosis(text):
    """diagnosis를 리스트 형태로 정규화"""
    if pd.isna(text):
        return []
    # 1) 개행, 탭, 중복 공백 정리
    text = re.sub(r'\s+', ' ', str(text)).strip()
    # 2) PRIMARY / SECONDARY / etc. 라벨 제거
    text = re.sub(r'PRIMARY|SECONDARY|PRIMARY DIAGNOSIS|SECONDARY DIAGNOSIS|DIAGNOSIS|Dx|:', 
                  '', text, flags=re.IGNORECASE)
    # 3) 번호 or bullet 처리 (- split by numbers "1.", "2)", "-", "•")
    parts = re.split(r'\d+[\.\)]\s*|[-•]\s*', text)
    # 4) 빈 항목 제거 & trim
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
    return parts


def extract_info(text: str):
    text = str(text)
    m = re.search(r'Gender:\s*(.*?),\s*Race:\s*(.*?),\s*Age:\s*(\d+)', text)
    if not m:
        return None, None, None  # 혹은 기본값 설정

    gender = m.group(1).strip()
    race   = m.group(2).strip()
    age    = int(m.group(3))
    start = (age // 10) * 10   # 54 → 50
    end   = start + 10         # 50 → 60

    return gender, race, f"{start}-{end}"


def llm_summarize_hpi(
    df, 
    pipe, 
    batch_size=8,
    hpi_col="HPI",
    diagnosis_col="diagnosis",
    info_col="patient_info"
):
    """
    DataFrame 전체 전처리:
    1) LLM HPI 요약(batch 방식)
    2) diagnosis 정규화
    3) patient_info 추출
    """

    # === 1. HPI 요약 프롬프트 ===
    prompts = [
        f"""
        You are a medical expert. Please provide ONLY a concise summary of the patient's History of Present Illness (HPI) in less than 100 words. 
        Do NOT include any additional text or explanations.

        Patient HPI: {text}

        Summary:"""
        for text in df[hpi_col].fillna("").astype(str).tolist()
    ]

    # === 2. Batch inference 실행 ===
    outputs = pipe(prompts, batch_size=batch_size, truncation=True)

    def extract_summary(g):
        try:
            text = g["generated_text"]
            summary = text.split("Summary:")[-1].strip()
            words = summary.split()
            return " ".join(words[:50])
        except:
            return ""

    df["llm_hpi_summary"] = [extract_summary(out) for out in outputs]
    # === 3. diagnosis 정규화 ===
    df["diagnosis_list"] = df[diagnosis_col].apply(normalize_diagnosis)
    # === 4. patient_info 추출 ===
    df["patient_info_extract"] = df[info_col].apply(extract_info)

    return df


# BM25 검색 함수들 (LLM 전처리 데이터 사용)
def bm25_search_diag_only(query_vec, rag_vectors, rag_df, k=15):
    sims = cosine_similarity(query_vec, rag_vectors).flatten()
    top_idx = np.argsort(sims)[-k:][::-1]
    # LLM 추출 진단명들 반환 (리스트)
    return [diag for sublist in rag_df.iloc[top_idx]['diagnosis_list'] for diag in sublist]

def bm25_search_full(query_vec, rag_vectors, rag_df, k=5):
    sims = cosine_similarity(query_vec, rag_vectors).flatten()
    top_idx = np.argsort(sims)[-k:][::-1]
    return rag_df.iloc[top_idx][['llm_hpi_summary', 'diagnosis_list']].to_dict('records')

def bm25_majority_vote(query_vec, rag_vectors, rag_df,):
    top_diags = bm25_search_diag_only(query_vec, rag_vectors, rag_df,)
    return Counter(top_diags).most_common(1)[0][0]


def summarize_single_hpi(text, pipe):
    prompt = f"""
        You are a medical expert. Provide ONLY a concise summary of the patient's History of Present Illness (HPI) in <50 words.

        Patient HPI: {text}

        Summary:
    """
    out = pipe(prompt, max_new_tokens=128, truncation=True)

    if isinstance(out, list):
        out = out[0]

    # dict or string 모두 처리
    if isinstance(out, dict):
        gen = out.get("generated_text", "")
    else:
        gen = str(out)

    summary = gen.split("Summary:")[-1].strip()
    words = summary.split()
    return " ".join(words[:50])


# 기존 후처리 함수
def validate_llm_output(llm_output):
    llm_lower = str(llm_output).lower().strip()
    invalid_keywords = ['unknown', 'cannot', 'insufficient', 'clinician', 'doctor', 
                       'diagnosis', 'diagnoses', 'primary diagnosis', 
                    #    'rationale'
                       ]
    if any(keyword in llm_lower for keyword in invalid_keywords):
        return None
    words = llm_lower.split()
    if len(words) < 2 or len(llm_lower) < 10:
        return None
    medical_keywords = ['fracture', 'pain', 'cholecystitis', 'appendicitis', 'cellulitis',
                       'pneumonia', 'obstruction', 'chf', 'copd', 'infarction']
    if not any(keyword in llm_lower for keyword in medical_keywords):
        return None
    return llm_output


def extract_diagnosis_from_llm_improved(text):
    text = str(text).lower()
    patterns = [
        r'(acute|chronic)\s+(appendicitis|cholecystitis|chf|cellulitis|fracture)',
        r'(hip|ankle|femur|tibia)\s+(fracture)',
        r'(abdominal|chest)\s+(pain)',
        r'\b(cellulitis|pneumonia|obstruction|cholelithiasis)\b'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group().strip()
            validated = validate_llm_output(candidate)
            if validated:
                return validated
    words = re.findall(r'\b[a-z]{5,}\b', text)
    for word in words:
        validated = validate_llm_output(word)
        if validated:
            return validated
    return None


# LLM 전처리 + 예측 진단명 하이브리드 함수
def hybrid_rag_llm_llm_preprocess(hpi_text, llm_pipe, vectorizer, rag_vectors, rag_df, threshold=CONFIDENCE_THRESHOLD, use_rag=True):
    """
    LLM 전처리 + 다중 후보 하이브리드
    ### use_rag 옵션 추가 
    """
    
    q_text = summarize_single_hpi(hpi_text, llm_pipe)  # 쿼리도 LLM 요약
    
    bm25_candidates = []
    bm25_diag = None
    context = ""

    #### RAG 사용 여부 ####
    if use_rag:
        # 1. BM25 검색 (LLM 전처리 데이터 사용)
        q_vec = vectorizer.transform([q_text])
        bm25_candidates = bm25_search_diag_only(q_vec, k=7)
        bm25_diag = bm25_majority_vote(q_vec, rag_vectors, rag_df,)
        # 2. LLM 컨텍스트 생성
        similar_cases = bm25_search_full(q_vec, rag_vectors, rag_df, k=3)
        context = "Similar cases:\n" + "\n".join([
            f"{i+1}. HPI: {case['llm_hpi_summary']} → Diags: {', '.join(case['diagnosis_list'])}" 
            for i, case in enumerate(similar_cases)
        ])
    else:
        # RAG 미사용 시 컨텍스트 비움
        context = "No similar cases provided. Diagnose based on HPI only."

    #### Rationale ####
    # For each diagnosis, provide a brief rationale explaining why.
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a skilled clinician. Provide up to 3 possible diagnoses from HPI.

        Patient HPI: {hpi_text}
        {context}
        Format each line as:
        Diagnoses: [Diagnoses list]


        
        Diagnoses:"""
            
    # 3. 진단명 예측
    llm_candidates = []
    # llm_rationales = [] # Rationale 저장

    try:
        llm_outputs = llm_pipe(prompt, num_return_sequences=3)
        for out in llm_outputs:
            generated_text = out['generated_text']
            # 진단명 추출 
            raw_diag = extract_diagnosis_from_llm_improved(generated_text)

            # # Rationale 추출 
            # # LLM이 포맷을 지켰다면 "-" 이후가 Rationale, 아니면 전체 텍스트
            # if "Rationale:" in generated_text:
            #     rationale_text = generated_text.split("Rationale:")[-1].strip()
            # else:
            #     rationale_text = generated_text.strip()

            if raw_diag and raw_diag != "unknown":
                llm_candidates.append(raw_diag)
                # llm_rationales.append(rationale_text)

        llm_candidates = list(set(llm_candidates))[:3]
        # rationales 개수 맞추기 (중복 제거 등으로 개수가 안 맞을 수 있으므로 단순화)
        if len(llm_rationales) > len(llm_candidates):
            llm_rationales = llm_rationales[:len(llm_candidates)]
            
    except:
        llm_candidates = ["unknown"]
        # llm_rationales = ["No rationale generated"]
    
    # 4. 하이브리드 앙상블 (RAG 미사용시 BM25 제외)
    if use_rag:
        hybrid_candidates = bm25_candidates + llm_candidates * 2
    else:
        hybrid_candidates = llm_candidates # RAG 껐을 땐 LLM만
        
    candidate_counts = Counter(hybrid_candidates)
    top_candidates = [c for c, _ in candidate_counts.most_common(5)][:3]
    
    # 후보가 비었을 경우 처리
    if not top_candidates:
        top_candidates = ["unknown"]

    # 5. 상위 3개 유사도 계산
    similarity_scores = [semantic_similarity(cand, hpi_text) for cand in top_candidates]
    
    # 6. 신뢰도 (최고 유사도 기준)
    confidence = max(similarity_scores) if similarity_scores else 0.0
    
    # 7. 상태 판단
    if confidence >= threshold:
        status, action, color = "🟢 HIGH_CONFIDENCE"
    elif confidence >= 0.45:
        status, action, color = "🟡 REVIEW_NEEDED"
    else:
        status, action, color = "🔴 LOW_CONFIDENCE"
    
    return {
        "diagnosis_candidates": top_candidates,
        # "rationales": llm_rationales, 
        "similarity_scores": similarity_scores,
        "bm25_candidates": bm25_candidates[:3],
        "llm_candidates": llm_candidates,
        "bm25_fallback": bm25_diag,
        "confidence": round(confidence, 3),
        "status": status,
        "action": action,
        "color": color,
        "hpi_summary": q_text,
        "hpi": hpi_text[:120] + "...",
        "use_rag": use_rag
    }


# == 11. LLM 일관성 테스트  (multi diagnoses) ==

def get_all_answers_multicandidate(records, model_func, args, llm_pipe, n: int = 5, use_rag: bool = True):
    vectorizer, rag_vectors, rag_df = args
    all_preds = []
    # all_rationales = []
    for hpi in records:
        preds_list = []
        # rats_list = []
        for _ in range(n):
            # use_rag 옵션
            result = model_func(hpi, llm_pipe, vectorizer, rag_vectors, rag_df, use_rag=use_rag)
            preds_list.append(result['diagnosis_candidates'])
            # rats_list.append(result.get('rationales', []))
        all_preds.append(preds_list)
        # all_rationales.append(rats_list)
    # return all_preds, all_rationales
    return all_preds

def calculate_set_similarity(list1, list2):
    """
    [Set-to-Set Similarity]
    두 리스트가 '구성적으로' 얼마나 유사한지 평가
    - 정답이 여러 개일 때, 모델이 그걸 다 커버했는지(Recall)
    - 모델의 예측들이 헛다리 안 짚고 정답과 관련 있는지(Precision)
    두 가지를 모두 고려하여 평균냄
    """
    if not list1 or not list2: return 0.0
    
    # 1. List1(기준) -> List2(타겟): "List1의 항목들을 List2가 얼마나 잘 커버했나?"
    scores_1_to_2 = []
    for t1 in list1:
        # t1과 가장 유사한 t2를 찾아서 점수 반영
        best_score = max([semantic_similarity(t1, t2) for t2 in list2]) if list2 else 0
        scores_1_to_2.append(best_score)
    
    # 2. List2(기준) -> List1(타겟): "List2의 항목들은 List1과 얼마나 관련있나?"
    scores_2_to_1 = []
    for t2 in list2:
        best_score = max([semantic_similarity(t2, t1) for t1 in list1]) if list1 else 0
        scores_2_to_1.append(best_score)
        
    # 양방향 평균 (Recall 성격 + Precision 성격)
    return (np.mean(scores_1_to_2) + np.mean(scores_2_to_1)) / 2.0


def evaluate_batch_multicandidate(records, trues, ids, model_func, args, pipe, n_repeat=5, use_rag=True):
    # 1. 먼저 5번 반복(n_repeat) 실행하여 결과 수집
    # batch_preds 구조: [ [Run1, Run2...], [Run1, Run2...] ... ] (피험자별로 묶여있음)
    # batch_preds, batch_rationales = get_all_answers_multicandidate(records, model_func, pipe, n=n_repeat, use_rag=use_rag)
    batch_preds = get_all_answers_multicandidate(records, model_func, args, pipe, n=n_repeat, use_rag=use_rag)
    
    results = {} # 최종 저장: { "ID_001": {결과}, "ID_002": {결과} ... }

    print("\n=== 🔄 CONSISTENCY TEST ===")
    
    # for i, (record_preds_list, record_rats_list, true_diag, patient_id, hpi_text) in tqdm(enumerate(zip(batch_preds, batch_rationales, trues, ids, records))):
    for i, (record_preds_list, true_diag, patient_id, hpi_text) in tqdm(enumerate(zip(batch_preds, trues, ids, records))):
        
        # 정답 포맷팅
        if not isinstance(true_diag, list):
            true_diag = [str(true_diag)]
        
        # --- [1] Accuracy (5회 평균) ---
        acc_scores = []
        for preds in record_preds_list: 
            # Set-to-Set 유사도 계산
            run_score = calculate_set_similarity(preds, true_diag)
            acc_scores.append(run_score)
        semantic_accuracy = float(np.mean(acc_scores))

        # --- [2] Consistency (5회 간 쌍 비교 평균) ---
        if len(record_preds_list) > 1:
            from itertools import combinations
            run_pairs = list(combinations(record_preds_list, 2))
            pair_scores = [calculate_set_similarity(p1, p2) for p1, p2 in run_pairs]
            semantic_consistency = float(np.mean(pair_scores))
        else:
            semantic_consistency = 1.0
            
        # --- [3] Uncertainty ---
        semantic_uncertainty = 1.0 - semantic_consistency

        # [저장] ID를 Key로 하여, 해당 피험자의 모든 정보를 하나의 Value로 저장
        results[patient_id] = {
            "patient_id": patient_id,
            "hpi": hpi_text[:200] + "..." if len(hpi_text) > 200 else hpi_text, # 원문 일부 저장
            "true_diagnosis": true_diag,
            # 5번의 결과 리스트
            "predictions_per_repeat": record_preds_list, 
            # "rationales_per_repeat": record_rats_list,
            # 지표 평균값
            "semantic_accuracy": semantic_accuracy,         
            "semantic_consistency": semantic_consistency,   
            "semantic_uncertainty": semantic_uncertainty,
            "use_rag": use_rag
        }
        
    return results





def load_main_llm(model_name):
    print("메인 LLM 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id
    return pipe




def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--rag", action="store_true", help="enable RAG")
    args = ap.parse_args()

    seed = 0
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

    # === 1. 데이터 로드 ===
    df = pd.read_csv(DATA_PATH)
    # 실제 ID 컬럼이 있다면 사용하고, 없다면 인덱스를 ID로 보존
    if 'stay_id' not in df.columns:
        df['stay_id'] = df.index 

    df = df.dropna(subset=['HPI','patient_info', 'diagnosis']).reset_index(drop=True)
    rag_df = df.sample(frac=0.8, random_state=0)
    test_df = df.drop(rag_df.index).reset_index(drop=True)


    # === 4. seperating data ===
    rag_df = llm_summarize_hpi(rag_df, llm_preprocess_pipe, batch_size=8)
    test_df = llm_summarize_hpi(test_df, llm_preprocess_pipe, batch_size=8)


    # === 5. 검색용 텍스트 (HPI 요약 + 진단명 합치기) ====
    rag_df['search_text'] = rag_df['llm_hpi_summary'] + ' ' + ' ' +  rag_df['patient_info_extract'].apply(
        lambda x: ' '.join([str(v) for v in x if v]) if isinstance(x, tuple) else str(x)
    ) + ' ' + rag_df['diagnosis_list'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    test_df['query_text'] = test_df['llm_hpi_summary']

    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
    rag_vectors = vectorizer.fit_transform(rag_df['search_text'])
    test_queries = vectorizer.transform(test_df['query_text'])

    print(f"✅ LLM 전처리 완료: {len(rag_df)} RAG / {len(test_df)} Test")


    # === 6. 메인 LLM 로드 (진단명 예측) ===
    print("메인 LLM 로드 중...")
    # model_name = "FreedomIntelligence/HuatuoGPT-o1-8B"
    model_name = args.model
    llm_pipe = load_main_llm(model_name)

    try:
        # === 7. Test 시작 ===
        test_subset = test_df 
        hpi_samples = test_subset['HPI'].tolist()
        true_samples = test_subset['diagnosis_list'].tolist()
        id_samples = test_subset['stay_id'].tolist() 

        # RAG 사용 여부 설정 (True or False)
        USE_RAG_OPTION = args.rag 

        results = evaluate_batch_multicandidate(
            hpi_samples, 
            true_samples, 
            id_samples, 
            hybrid_rag_llm_llm_preprocess, 
            args=(vectorizer, rag_vectors, rag_df),
            pipe=llm_pipe,
            n_repeat=5, 
            use_rag=USE_RAG_OPTION
        )

    finally:
        print("=" * 80)
        # output 예시
        for idx in list(results.keys())[:5]:
            res = results[idx]
            print(f"   Patient ID: {res['patient_id']}")
            print(f"   True: {res['true_diagnosis']}")
            print(f"   🔹 Accuracy: {res['semantic_accuracy']:.3f}")
            print(f"   🔹 Consistency: {res['semantic_consistency']:.3f}")
            print(f"   🔹 Uncertainty: {res['semantic_uncertainty']:.3f}")
            print(f"   Prediction: {res['predictions_per_repeat'][0]}")
            # if res['rationales_per_repeat'][0]:
            #     print(f"   💡 Rationale: {res['rationales_per_repeat'][0][0][:100]}...")
            print()

        M = model_name.split('/')[-1].replace("-", "_")
        if args.rag==True:
            rag = 'rag'
        else:
            rag = 'base'

        import os
        savedir = "/home/work/.dahyoun/class/text_ai/project/final_result"
        os.makedir(savedir, exist_ok=True)
        output_path = f"{savedir}/{M}_{rag}_results.json"

        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"💾 JSON 저장 완료: {output_path}")




if __name__ == "__main__":
    main()

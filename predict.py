# filename: predict.py
import os
import json
import time
import re
import pickle
import requests
import numpy as np
import pandas as pd
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================================
# 1. CONFIGURATION & AUTH
# ============================================================================

# Lấy Token từ biến môi trường (Docker) hoặc điền trực tiếp nếu test local
# QUAN TRỌNG: Hãy đảm bảo các biến này được set trong Docker hoặc điền vào đây.
AUTH_TOKEN = os.environ.get('API_TOKEN', "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0cmFuc2FjdGlvbl9pZCI6IjQ0N2IyOTY1LWEzNjQtNGNmNi05NTlkLTVjODc5MzliYTQwZSIsInN1YiI6ImY3NTczZmU0LWQxMjktMTFmMC1hNzY5LTY1YzE1OThiZjNhZCIsImF1ZCI6WyJyZXN0c2VydmljZSJdLCJ1c2VyX25hbWUiOiJkYmJhbzIzQGNsYy5maXR1cy5lZHUudm4iLCJzY29wZSI6WyJyZWFkIl0sImlzcyI6Imh0dHBzOi8vbG9jYWxob3N0IiwibmFtZSI6ImRiYmFvMjNAY2xjLmZpdHVzLmVkdS52biIsInV1aWRfYWNjb3VudCI6ImY3NTczZmU0LWQxMjktMTFmMC1hNzY5LTY1YzE1OThiZjNhZCIsImF1dGhvcml0aWVzIjpbIlVTRVIiLCJUUkFDS18yIl0sImp0aSI6ImZkOGFmOWMzLTBlYjMtNGEwOS1iY2Y2LWU4NzhhMDBkM2ZlMyIsImNsaWVudF9pZCI6ImFkbWluYXBwIn0.yRvRNQsk5hFIsx5wU9pHq7MTIiKwldYn6byfHbp3CY2dLjW7RLM7d_9I9TjnyOAFpGrNdevxkTwYZPJ1usl4_R9VQXSu1qNPTSIO4J88ok9pATC70A--MKwPrOivXrCPPxRDWFoc_ma97N625FCd8hGj5r_kVBj1hXrnyiITI_3FFubx7dy_1YnPAZk78IByXgIb8Gdze6aXDjClUC8aMFUg9y8b4jopnx2jThCwhmMDnMCcn4JNJ8nZChktAKfQx22GyJuogmKeh8ABswZ0xzMFV4W3vnxYJ_pOR_tjzWrcmw8bGqd1JY6v6jbrcFaKqG1irspGVOsdeN3luuU_4Q")
EMBED_ID = os.environ.get('EMBED_ID', "4525a834-464c-6f29-e063-62199f0a0f81")
EMBED_KEY = os.environ.get('EMBED_KEY', "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAIRkyCNeZKmtYJDygOze1L/dAZGk49s08rEhrMDvRwBgIqa6fD9bvJB4zVAhzRIzBRqIOsRoHfdr5Rg/P5UCALcCAwEAAQ==")
SMALL_ID = os.environ.get('SMALL_ID', "4525a834-464b-6f29-e063-62199f0a0f81")
SMALL_KEY = os.environ.get('SMALL_KEY', "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAIbFyYdkIUtluh8uQFrXcGTGXY4N/o/t6JLwNhL1VMTxhy95E3lerbsJJnDHdnmR6V2XBip9hfTryIZd2HeZvFMCAwEAAQ==")
LARGE_ID = os.environ.get('LARGE_ID', "4525a832-9105-15b3-e063-62199f0a3f23")
LARGE_KEY = os.environ.get('LARGE_KEY', "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAI3YMOIlnZ2O36pb0embkYcZpm0aNrWMt/PzqUEWs/WR+AbYMa7xWON+VbWFQFcrzV2io55PDJP6CtcRGKAWU6MCAwEAAQ==")

API_CONFIG = {
    'embed': {
        'url': "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding",
        'id': EMBED_ID,
        'key': EMBED_KEY
    },
    'small': {
        'url': "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small",
        'id': SMALL_ID,
        'key': SMALL_KEY
    },
    'large': {
        'url': "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large",
        'id': LARGE_ID,
        'key': LARGE_KEY
    }
}

# Cấu hình Path (Chuẩn Docker: Assets nằm trong /app/assets)
# Lưu ý: Trong Dockerfile bạn phải COPY assets vào đây
BASE_ASSETS_DIR = "/code/assets"

SAFETY_INDEX_PATH = f"{BASE_ASSETS_DIR}/Safety.index"
SAFETY_PKL_PATH = f"{BASE_ASSETS_DIR}/Safety.pkl"
MATHLOGIC_INDEX_PATH = f"{BASE_ASSETS_DIR}/vnpt_mathlogic_knowledge.index"
MATHLOGIC_PKL_PATH = f"{BASE_ASSETS_DIR}/vnpt_mathlogic_docstore.pkl"
GENERAL_INDEX_PATH = f"{BASE_ASSETS_DIR}/vnpt_knowledge_MERGED.index"
GENERAL_PKL_PATH = f"{BASE_ASSETS_DIR}/vnpt_knowledge_MERGED.pkl"

# INPUT/OUTPUT PATHS (Theo quy định BTC)
# Input thường được mount vào /data/private_test.json
INPUT_FILE_PATH = "/code/private_test.json"
OUTPUT_FILE_PATH = "/code/submission.csv"

SAFETY_THRESHOLD = 0.35
DELAY_LARGE = 0.1 # Giảm delay trong Docker vì chạy server
DELAY_SMALL = 0.1

# ============================================================================
# 2. API HELPER FUNCTIONS
# ============================================================================

def get_header(service_type):
    cfg = API_CONFIG[service_type]
    token_str = AUTH_TOKEN if "Bearer" in AUTH_TOKEN else f"Bearer {AUTH_TOKEN}"
    return {
        "Authorization": token_str,
        "Token-id": cfg['id'],
        "Token-key": cfg['key'],
        "Content-Type": "application/json"
    }

def get_embedding(text):
    headers = get_header('embed')
    payload = {"model": "vnptai_hackathon_embedding", "input": text, "encoding_format": "float"}
    for _ in range(3):
        try:
            resp = requests.post(API_CONFIG['embed']['url'], headers=headers, json=payload, timeout=5)
            if resp.status_code == 200: return resp.json()['data'][0]['embedding']
            elif resp.status_code == 429: time.sleep(1)
        except: time.sleep(1)
    return None

def call_llm_small(prompt):
    headers = get_header('small')
    payload = {
        "model": "vnptai_hackathon_small",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_completion_tokens": 100,
        "top_p": 0.95
    }
    try:
        resp = requests.post(API_CONFIG['small']['url'], headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content'].strip()
        else:
            return "GENERAL"
    except Exception as e:
        print(f"Small LLM Error: {e}")
        return "GENERAL"

def call_llm_large_v2(system_prompt, user_prompt, temperature=0.1):
    headers = get_header('large')
    payload = {
        "model": "vnptai_hackathon_large",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_completion_tokens": 512,
        "top_p": 0.95
    }
    try:
        resp = requests.post(API_CONFIG['large']['url'], headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            return re.sub(r"``````", "", content).strip()
        else:
            print(f"Large LLM Error {resp.status_code}")
            return None
    except Exception as e:
        print(f"Large LLM Exception: {e}")
        return None

# ============================================================================
# 3. KNOWLEDGE BASE CLASS
# ============================================================================

class KnowledgeBase:
    def __init__(self, name, index_path, pkl_path):
        print(f"Loading DB: {name}...")
        try:
            self.index = faiss.read_index(index_path)
            with open(pkl_path, "rb") as f:
                self.docs = pickle.load(f)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            self.index = None
            self.docs = []

    def search(self, vector, k=5):
        if self.index is None: return []
        distances, indices = self.index.search(vector, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.docs):
                results.append({
                    "content": self.docs[idx].get('content', ''),
                    "score": float(distances[0][i])
                })
        return results

# ============================================================================
# 4. TECHNIQUE & PIPELINE HELPERS
# ============================================================================

FEWSHOT_EXAMPLES = [
    {
        "question": "Phương trình bậc hai x² + 2x - 3 = 0 có nghiệm là gì?",
        "context": "Phương trình bậc hai ax² + bx + c = 0 có công thức nghiệm: x = (-b ± √Δ) / 2a, với Δ = b² - 4ac",
        "choices": {"A": "x = 1 hoặc x = -3", "B": "x = 2 hoặc x = -1", "C": "x = 3 hoặc x = -1", "D": "Vô nghiệm"},
        "correct_answer": "A",
        "reasoning": "Step 1: Đây là PT bậc 2 với a=1, b=2, c=-3\nStep 2: Δ = 16 > 0\nStep 3: x = 1 hoặc x = -3\nStep 4: Chọn A"
    },
    {
        "question": "Thủ đô của Pháp là thành phố nào?",
        "context": "Paris là thủ đô của nước Pháp...",
        "choices": {"A": "Lyon", "B": "Paris", "C": "Marseille", "D": "Nice"},
        "correct_answer": "B",
        "reasoning": "Step 1: Hỏi thủ đô Pháp\nStep 2: Context nói Paris\nStep 3: Loại trừ\nStep 4: Chọn B"
    }
]

def format_fewshot_prompt():
    text = "="*60 + "\nVÍ DỤ MẪU:\n" + "="*60 + "\n\n"
    for i, ex in enumerate(FEWSHOT_EXAMPLES, 1):
        text += f"EXAMPLE {i}:\nQ: {ex['question']}\nContext: {ex['context']}\nReasoning: {ex['reasoning']}\n✓ Ans: {ex['correct_answer']}\n\n"
    return text

def calculate_adaptive_temperature(question, category):
    base_temp = 0.05 if "MATH" in category.upper() else 0.12
    word_count = len(question.split())
    if word_count > 60: base_temp += 0.05
    elif word_count < 15: base_temp -= 0.02
    return min(max(base_temp, 0.0), 0.3)

def get_optimized_system_prompt(category, last_label, fewshot_text):
    fallback_instruction = "Ưu tiên dùng Context. TUY NHIÊN, nếu Context không đủ, HÃY SỬ DỤNG KIẾN THỨC CỦA BẠN. KHÔNG ĐƯỢC TRẢ LỜI LÀ KHÔNG BIẾT."
    return f"""{fewshot_text}
HƯỚNG DẪN GIẢI:
{'='*60}
{fallback_instruction}

Giải quyết theo 4 bước (Hiểu -> Tìm -> Tính -> Kết luận).
BẮT BUỘC RETURN JSON:
{{
  "step1": "...",
  "step2": "...",
  "step3": "...",
  "step4": "...",
  "answer": "X" # A-{last_label}
}}
"""

# --- Techniques ---

def rerank_documents(docs, question, small_llm_func, max_docs_to_keep=5):
    if len(docs) <= max_docs_to_keep: return docs
    docs_text = "\n".join([f"[Doc {i+1}] {d['content'][:200]}..." for i, d in enumerate(docs)])
    prompt = f"Rank docs for question: {question}\nDocs:\n{docs_text}\nReturn indices (e.g., 1, 3, 2):"
    try:
        out = small_llm_func(prompt)
        indices = [int(x)-1 for x in re.findall(r'\d+', out)]
        reranked = [docs[i] for i in indices if 0 <= i < len(docs)]
        return reranked[:max_docs_to_keep] if reranked else docs[:max_docs_to_keep]
    except: return docs[:max_docs_to_keep]

def generate_query_variations(question, small_llm_func):
    prompt = f"Rewrite this question in 2 ways for search (separated by |): {question}"
    try:
        txt = small_llm_func(prompt)
        return [v.strip() for v in txt.split('|') if v.strip()][:2]
    except: return []

def search_with_expansion(question, db, small_llm_func, k_per_query=3):
    all_docs = []
    q_vec = get_embedding(question)
    if q_vec:
        all_docs.extend(db.search(np.array([q_vec]).astype('float32'), k=k_per_query+2))
    
    if len(question.split()) < 20:
        for v in generate_query_variations(question, small_llm_func):
            v_vec = get_embedding(v)
            if v_vec: all_docs.extend(db.search(np.array([v_vec]).astype('float32'), k=k_per_query))
            
    unique, seen = [], set()
    for d in all_docs:
        sig = d['content'][:100]
        if sig not in seen:
            seen.add(sig)
            unique.append(d)
    return unique

def ensemble_retrieval(question, db_math, db_general, small_llm_func, k_math=5, k_general=5):
    general_docs = search_with_expansion(question, db_general, small_llm_func, k_per_query=k_general)
    math_docs = []
    q_vec = get_embedding(question)
    if q_vec: math_docs = db_math.search(np.array([q_vec]).astype('float32'), k=k_math)
    
    combined = general_docs + math_docs
    unique, seen = [], set()
    for d in combined:
        sig = d['content'][:100]
        if sig not in seen:
            seen.add(sig)
            unique.append(d)
    return unique

def self_correct_answer(original_ans, original_reasoning, question, choices_text, context_text, large_llm_func):
    prompt = f"""
Reflect on this answer.
Q: {question}
Choices: {choices_text}
Context: {context_text[:1500]}
Your Answer: {original_ans} (Reason: {original_reasoning})
If wrong, correct it. Return JSON: {{ "final_answer": "X", "critique": "..." }}
"""
    out = large_llm_func("You are a critique.", prompt, temperature=0.0)
    try:
        res = json.loads(out)
        return res.get('final_answer', original_ans), res.get('critique', '')
    except: return original_ans, "Correction Failed"

# ============================================================================
# 5. RAG PIPELINE ULTIMATE CLASS
# ============================================================================

class RAGPipelineUltimate:
    def __init__(self):
        print("Initializing Ultimate Pipeline...")
        self.db_safety = KnowledgeBase("Safety", SAFETY_INDEX_PATH, SAFETY_PKL_PATH)
        self.db_math = KnowledgeBase("Math", MATHLOGIC_INDEX_PATH, MATHLOGIC_PKL_PATH)
        self.db_general = KnowledgeBase("General", GENERAL_INDEX_PATH, GENERAL_PKL_PATH)
        print("✓ Knowledge Bases Loaded.")

    def process_question(self, q_item, enable_reranking=True, enable_self_correction=True):
        qid = q_item.get('qid', 'unknown')
        question = q_item['question']
        choices = q_item['choices']

        choices_map = {chr(65+i): c for i, c in enumerate(choices)}
        valid_labels = list(choices_map.keys())
        last_label = valid_labels[-1] if valid_labels else 'A'
        choices_text = "\n".join([f"{k}. {v}" for k, v in choices_map.items()])

        # --- PHASE 1: SAFETY ---
        q_vec = get_embedding(question)
        if q_vec:
            q_vec_np = np.array([q_vec]).astype('float32')
            hits = self.db_safety.search(q_vec_np, k=1)
            if hits and hits[0]['score'] < SAFETY_THRESHOLD:
                return {"qid": qid, "answer": "A"} # Safety Trigger -> Default A

        # --- PHASE 2: RETRIEVAL ---
        docs = ensemble_retrieval(question, self.db_math, self.db_general, call_llm_small)

        # --- PHASE 3: RERANKING ---
        if enable_reranking and len(docs) > 7:
            docs = rerank_documents(docs, question, call_llm_small, max_docs_to_keep=7)

        # --- PHASE 4: CONTEXT ---
        context_text = "\n\n---\n\n".join([f"[DOC {i+1}]\n{d['content']}" for i, d in enumerate(docs)])
        if len(context_text) > 60000: context_text = context_text[:60000]

        # --- PHASE 5: GENERATION ---
        category = "MATH" if any(kw in question.lower() for kw in ["tính", "phương trình"]) else "GENERAL"
        fewshot_text = format_fewshot_prompt()
        sys_msg = get_optimized_system_prompt(category, last_label, fewshot_text)
        temp = calculate_adaptive_temperature(question, category)
        user_msg = f"INFO:\n{context_text}\n\nQ:\n{question}\n\nCHOICES:\n{choices_text}\n\nSelect the best answer."

        llm_out = call_llm_large_v2(sys_msg, user_msg, temperature=temp)
        
        # Parsing Logic V4 (Hybrid)
        final_ans, reason = "A", "Parse Failed"
        if llm_out:
            try:
                s = llm_out.find('{'); e = llm_out.rfind('}')
                if s != -1 and e != -1:
                    res = json.loads(llm_out[s:e+1])
                    final_ans = res.get('answer', 'A')
                    reason = res.get('step4', '') or str(res)
                else: raise ValueError()
            except:
                match = re.search(r'answer["\']?\s*[:=]\s*["\']?([A-Z])', llm_out, re.IGNORECASE)
                if match: final_ans = match.group(1).upper()
                reason = "Regex Extracted"

        # --- PHASE 6: SELF-CORRECTION ---
        is_math = "MATH" in category.upper()
        if enable_self_correction and (is_math or len(reason) < 50):
            new_ans, critique = self_correct_answer(final_ans, reason, question, choices_text, context_text, call_llm_large_v2)
            if new_ans in valid_labels: final_ans = new_ans

        # --- FALLBACK FORCE CHOICE ---
        if final_ans not in valid_labels:
            out = call_llm_large_v2("You are a quiz machine.", f"Question: {question}\n{choices_text}\nPick one letter (A,B,C,D). No explanation.", 0.1)
            m = re.search(r'\b([A-Z])\b', str(out))
            if m and m.group(1) in valid_labels: final_ans = m.group(1)
            else: final_ans = 'A' # Last resort

        return {"qid": qid, "answer": final_ans}

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    print("Starting RAG Pipeline Prediction...")
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Input file not found at {INPUT_FILE_PATH}. Using dummy data.")
        data = [{"qid": "test_1", "question": "1+1=?", "choices": ["1", "2", "3", "4"]}]
    else:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # 2. Run Pipeline
    pipeline = RAGPipelineUltimate()
    results = []
    
    # Using ThreadPool for concurrency (Optional, be careful with rate limits)
    # Set max_workers=1 to be safe, or higher if API allows
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(pipeline.process_question, item): item for item in data}
        for future in tqdm(as_completed(futures), total=len(data)):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Error processing item: {e}")
                results.append({"qid": futures[future]['qid'], "answer": "A"})

    # 3. Export CSV
    df = pd.DataFrame(results)
    
    # Sort numeric qid if possible
    try:
        df['sort'] = df['qid'].apply(lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else 9999)
        df = df.sort_values('sort').drop('sort', axis=1)
    except: pass
    
    # Ensure correct columns
    df = df[['qid', 'answer']]
    df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Submission saved to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()

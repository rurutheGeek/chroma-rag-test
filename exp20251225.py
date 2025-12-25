import os
import time
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
import psutil
import pandas as pd
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import PersistentClient
from tqdm import tqdm

# --- 実験設定 ---
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
CHROMA_PATH = ".chroma_experiment"
COLLECTION_NAME = "eval_collection"

M_LIST = [4, 8, 16, 32, 48]
EF_CONSTRUCTION_LIST = [20, 100, 300]
EF_SEARCH_LIST = [10] 

# --- 1. ユーティリティ関数 ---

def get_dir_size(path):
    total = 0
    if not os.path.exists(path): return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)

def get_peak_memory():
    p = psutil.Process(os.getpid())
    mem = getattr(p.memory_info(), 'peak_wset', p.memory_info().rss)
    return mem / (1024 * 1024)

def load_and_split(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return []
    loaders = [
        DirectoryLoader(folder, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(folder, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    ]
    docs = []
    for loader in loaders: docs.extend(loader.load())
    if not docs: return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)

def get_ground_truth(query_vec, all_vectors, k=10):
    q = query_vec / np.linalg.norm(query_vec)
    v = all_vectors / np.linalg.norm(all_vectors, axis=1)[:, np.newaxis]
    similarities = np.dot(v, q)
    return set(np.argsort(similarities)[-k:][::-1])

# --- 2. グラフ出力関数 ---

def plot_comprehensive_results(results, timestamp):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # 1. Recall vs Latency (M固定で結ぶ)
    for m in M_LIST:
        sub = [r for r in results if r['m'] == m]
        if not sub: continue
        sub.sort(key=lambda x: x['ef_c'])
        recalls = [x['recall'] for x in sub]
        latencies = [x['latency'] for x in sub]
        
        line, = axes[0].plot(recalls, latencies, marker='o', label=f"M={m}")
        # 点の近くにef_cを注釈として表示
        for x in sub:
            axes[0].annotate(f"ef_c={x['ef_c']}", (x['recall'], x['latency']), 
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    axes[0].set_xlabel("Recall@10")
    axes[0].set_ylabel("Avg Latency (ms)")
    axes[0].set_title("Recall vs Latency Performance (by M)")
    axes[0].set_xlim(right=1.0)
    axes[0].legend(fontsize='x-small')
    axes[0].grid(True, alpha=0.3)

    # 2. Index Size vs Max Recall (M固定で結ぶ)
    for m in M_LIST:
        sub = [r for r in results if r['m'] == m]
        if not sub: continue
        sub.sort(key=lambda x: x['ef_c'])
        sizes = [x['index_size'] for x in sub]
        recalls = [x['recall'] for x in sub]
        
        line, = axes[1].plot(sizes, recalls, marker='o', label=f"M={m}")
        for x in sub:
            axes[1].annotate(f"ef_c={x['ef_c']}", (x['index_size'], x['recall']), 
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    axes[1].set_xlabel("Index Size on Disk (MB)")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Storage Efficiency (Index Size vs Recall)")
    axes[1].set_ylim(bottom=None, top=1.0)
    axes[1].legend(fontsize='x-small')
    axes[1].grid(True, alpha=0.3)

    # 3. Connectivity (既存機能維持)
    for ef_c in EF_CONSTRUCTION_LIST:
        sub_m = []
        recalls = []
        for m in M_LIST:
            match = [r for r in results if r['m'] == m and r['ef_c'] == ef_c]
            if match:
                sub_m.append(m)
                recalls.append(max([x['recall'] for x in match]))
        if sub_m:
            axes[2].plot(sub_m, recalls, marker='s', label=f"ef_c={ef_c}")
    
    axes[2].set_xlabel("Parameter M (Connectivity)")
    axes[2].set_ylabel("Achieved Recall")
    axes[2].set_title("Graph Connectivity Analysis")
    axes[2].set_ylim(bottom=None, top=1.0)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{timestamp}_hnsw_disk_efficiency_eval.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    print(f"\nGraphs saved as '{filepath}'")

# --- 3. 実験メインロジック ---

def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    
    print("="*50)
    print(f"[DEVICE INFO] Running on: {device.upper()}")
    if has_cuda:
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
    print("="*50)

    embeddings_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": device}
    )

    docs = load_and_split(INPUT_DIR)
    if not docs: return
    
    texts = [d.page_content for d in docs]
    total_docs = len(texts)
    
    batch_size = 64 if not has_cuda else 256
    all_vectors = []
    
    print(f"Generating embeddings for {total_docs} documents...")
    for i in tqdm(range(0, total_docs, batch_size), desc="Embedding Progress"):
        batch_texts = texts[i : i + batch_size]
        batch_vectors = embeddings_model.embed_documents(batch_texts)
        all_vectors.extend(batch_vectors)
    
    vectors = np.array(all_vectors)
    ids = [f"id_{i}" for i in range(total_docs)]

    results_data = []
    client = None

    for m in M_LIST:
        for ef_c in EF_CONSTRUCTION_LIST:
            if client is not None:
                del client
                gc.collect()
                time.sleep(1)

            if os.path.exists(CHROMA_PATH):
                for _ in range(5):
                    try:
                        shutil.rmtree(CHROMA_PATH)
                        break
                    except:
                        time.sleep(2)

            client = PersistentClient(path=CHROMA_PATH)
            
            try:
                existing_cols = client.list_collections()
                if any(col.name == COLLECTION_NAME for col in existing_cols):
                    client.delete_collection(COLLECTION_NAME)
                    time.sleep(0.5)
            except Exception: pass
            
            collection = client.create_collection(
                name=COLLECTION_NAME,
                configuration={
                    "hnsw": {
                        "space": "cosine",
                        "max_neighbors": int(m),
                        "ef_construction": int(ef_c)
                    }
                }
            )

            print(f"\n[BUILD] M={m}, ef_c={ef_c}...")
            chroma_batch_size = 5000 
            for i in range(0, len(vectors), chroma_batch_size):
                end_idx = i + chroma_batch_size
                collection.add(
                    embeddings=vectors[i:end_idx].tolist(),
                    documents=texts[i:end_idx],
                    ids=ids[i:end_idx]
                )
            
            index_size = get_dir_size(CHROMA_PATH)

            sample_size = min(20, len(vectors))
            sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
            
            for ef_s in EF_SEARCH_LIST:
                latencies = []
                recalls = []
                
                for idx in sample_indices:
                    query_vec = vectors[idx]
                    gt_ids = get_ground_truth(query_vec, vectors, k=10)
                    
                    t_start = time.perf_counter()
                    query_res = collection.query(query_embeddings=[query_vec.tolist()], n_results=10)
                    latencies.append((time.perf_counter() - t_start) * 1000)

                    retrieved_ids = {int(res_id.split('_')[1]) for res_id in query_res['ids'][0]}
                    recall = len(gt_ids.intersection(retrieved_ids)) / 10
                    recalls.append(recall)

                avg_latency = sum(latencies) / len(latencies)
                avg_recall = sum(recalls) / len(recalls)
                efficiency = avg_recall / max(index_size, 0.1)
                
                print(f"M:{m}, ef_c:{ef_c}, recall:{avg_recall:.2f}, size:{index_size:.2f}MB, eff:{efficiency:.4f}")
                
                results_data.append({
                    'm': m, 'ef_c': ef_c, 'ef_s': ef_s,
                    'latency': avg_latency, 'recall': avg_recall,
                    'index_size': index_size, 'efficiency': efficiency
                })

    # CSV出力
    df = pd.DataFrame(results_data)
    csv_filename = f"{timestamp}_experiment_results.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"\nCSV data saved as '{csv_path}'")

    analyze_optimal_m(results_data)
    plot_comprehensive_results(results_data, timestamp)

def analyze_optimal_m(results):
    summary = {}
    for r in results:
        key = (r['m'], r['ef_c'])
        if key not in summary or r['efficiency'] > summary[key]['efficiency']:
            summary[key] = r

    sorted_efficiency = sorted(summary.values(), key=lambda x: x['efficiency'], reverse=True)
    
    print("\n" + "="*50)
    print("--- Storage Efficiency Ranking (Recall per MB) ---")
    for entry in sorted_efficiency:
        print(f"M={entry['m']}, ef_c={entry['ef_c']} -> Efficiency: {entry['efficiency']:.4f}")
    
    best = sorted_efficiency[0]
    print("\n[Recommendation]")
    print(f"The most resource-efficient configuration is M={best['m']}, ef_c={best['ef_c']}.")
    print("Reason: This setup provides the highest recall relative to its storage footprint.")
    print("="*50)

if __name__ == "__main__":
    run_experiment()
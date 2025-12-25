from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA # 非推奨
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import torch
import dateutil.parser
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
import japanize_matplotlib
from shared_logger import set_logger, get_logger

# ---------------------------------------------------------
# Rust版対応 計測モンキーパッチ (L1用)
# ---------------------------------------------------------
import chromadb.api.models.Collection

# 1. Collection.add (表層: 受付)
target_collection = chromadb.api.models.Collection.Collection
original_collection_add = target_collection.add

def patched_collection_add(self, *args, **kwargs):
    logger_instance = get_logger()
    count = 0
    if "ids" in kwargs: count = len(kwargs["ids"])
    elif len(args) > 0: count = len(args[0])

    if logger_instance:
        # layer="L1" を指定してファイルに書き込む
        with logger_instance.log_event(f"ChromaDB登録 (batch {count})", layer="L1"):
            return original_collection_add(self, *args, **kwargs)
    else:
        return original_collection_add(self, *args, **kwargs)

target_collection.add = patched_collection_add
print(f"[INFO] Patch applied: Collection.add (L1 Logging)")


import csv

class TimelineLogger:
    def __init__(self, log_file="profile_log.csv"):
        self.records = []
        self.log_file = log_file
        
        # 起動時にログファイルを初期化（空にする）
        with open(self.log_file, "w", newline="") as f:
            pass # create new empty file

    @contextmanager
    def log_event(self, name: str, layer: str = "L1"):
        """
        L1 (Python) の計測を行い、CSVに即座に書き込む
        """
        print(f"[{name}] 開始")
        # Rustと合わせるため time.time() (UNIX時間) を使用
        start_time = time.time() 
        yield
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"[{name}] 完了 ({duration:.4f}秒)")

        # メモリにも残すが、CSVにも書く
        self.records.append((name, start_time, end_time))
        
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([layer, name, f"{start_time:.6f}", f"{end_time:.6f}"])

    def load_merged_records(self):
        """
        CSVからRust(L2-L4)のデータも含めて全データを読み込む
        """
        merged_records = []
        if not os.path.exists(self.log_file):
            return self.records

        with open(self.log_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                layer, name, start, end = row
                # (表示名, 開始, 終了) のタプルにする
                # グラフで見やすくするため、名前にレイヤーを付与
                display_name = f"[{layer}] {name}"
                merged_records.append((display_name, float(start), float(end)))
        
        return merged_records

# ---------------------------------------------------------
# 修正版: PerformanceCallbackHandler
# ---------------------------------------------------------
class PerformanceCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.llm_start_time = 0
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        # ロガーに合わせて perf_counter を使用
        self.llm_start_time = time.perf_counter()
        print(f"[LLM 回答生成] 開始")

    def on_llm_end(self, response, **kwargs):
        end_time = time.perf_counter()
        duration = end_time - self.llm_start_time
        print(f"[LLM 回答生成] 完了 ({duration:.4f}秒)")

        # 共通ロガーを取得してレコードに直接追加
        logger_instance = get_logger()
        if logger_instance:
            # (イベント名, 開始, 終了) の形式で追加
            logger_instance.records.append(("LLM 回答生成", self.llm_start_time, end_time))

# ---------------------------------------------------------
# 修正版: MeasuredEmbeddings
# ---------------------------------------------------------
class MeasuredEmbeddings:
    def __init__(self, wrapped_embeddings):
        self.wrapped_embeddings = wrapped_embeddings

    def embed_documents(self, texts):
        logger_instance = get_logger()
        # ドキュメント登録時の埋め込み計測
        if logger_instance:
            # 詳細分析のためバッチサイズも記録
            with logger_instance.log_event(f"ドキュメント埋め込み (batch {len(texts)})"):
                return self.wrapped_embeddings.embed_documents(texts)
        else:
            return self.wrapped_embeddings.embed_documents(texts)

    def embed_query(self, text):
        logger_instance = get_logger()
        # 検索クエリ埋め込み時（ここを計測）
        if logger_instance:
            with logger_instance.log_event("クエリ埋め込み"):
                return self.wrapped_embeddings.embed_query(text)
        else:
            # ロガーがない場合のフォールバック（printのみ）
            start = time.perf_counter()
            result = self.wrapped_embeddings.embed_query(text)
            end = time.perf_counter()
            print(f"[クエリ埋め込み] 完了 ({end - start:.4f}秒)")
            return result
        


def plot_timeline(records: list, output_filename: str = "timeline.png"):
    """
    (イベント名, 開始時刻, 終了時刻) のリストから
    タイムライン画像を生成・保存する
    """
    if not records:
        print("描画するデータがありません。")
        return
    
    # データから描画用の要素を抽出
    base_time = min(rec[1] for rec in records)
    event_names = [rec[0] for rec in records]
    start_times = [rec[1] - base_time for rec in records]
    end_times = [rec[2] - base_time for rec in records]
    
    # 各イベントの処理時間（棒の長さ）を計算
    durations = [end - start for start, end in zip(start_times, end_times)]

    # --- 色の決定ロジック (追加) ---
    colors = []
    for name in event_names:
        if "[L1]" in name:
            colors.append('#1f77b4') # 青 (Python)
        elif "[L2]" in name:
            colors.append('#ff7f0e') # オレンジ (Rust Binding)
        elif "[L3]" in name:
            colors.append('#2ca02c') # 緑 (Frontend)
        elif "[L4]" in name:
            colors.append('#d62728') # 赤 (Engine)
        else:
            colors.append('#d3d3d3') # グレー

    # 描画処理
    fig, ax = plt.subplots(figsize=(12, len(event_names) * 0.6 + 1))

    # 横棒グラフを描画 (color引数にリストを渡して色分け)
    bars = ax.barh(event_names, durations, left=start_times, height=0.6, color=colors)

    # 数値ラベルの追加
    for bar, duration in zip(bars, durations):
        width = bar.get_width()
        # 0.00001秒単位まで表示
        label_text = f" {duration:.5f} s"
        ax.text(bar.get_x() + width, bar.get_y() + bar.get_height()/2, label_text, 
                va='center', fontsize=9, color='black')

    # 見た目を整える
    ax.invert_yaxis()  # 最初のイベントが上に来るようにする
    ax.set_xlabel("時間 (秒)")
    ax.set_title("処理タイムライン")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # ラベルが見切れないようにX軸の範囲を調整
    max_end = max([s + d for s, d in zip(start_times, durations)])
    ax.set_xlim(0, max_end * 1.15)

    # 画像を保存
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"'{output_filename}' として画像を保存しました。")
    

def convert_records_to_relative(records: list) -> list:
    """
    TimelineLoggerのrecords（絶対時刻）を
    プロット用のlogged_data（相対時刻）に変換する。
    """
    if not records:
        return []
        
    # 1. 全体の開始時刻（最も早い開始時刻）を見つける
    global_start_time = min(rec[1] for rec in records)

    # 2. 各時刻を全体の開始時刻からの差分（相対時間）に変換する
    logged_data = []
    for name, start_abs, end_abs in records:
        relative_start = start_abs - global_start_time
        relative_end = end_abs - global_start_time
        logged_data.append((name, relative_start, relative_end))
        
    return logged_data


# 日付文字列をパースする関数
def parse_twitter_date(date_str):
    try:
        return dateutil.parser.parse(date_str)
    except:
        return None

# 1. DeepSeek API 用カスタム LLM
class DeepSeekAPI(LLM):
    api_key: str
    model_name: str

    def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
        super().__init__(api_key=api_key, model_name=model_name)  # ここを修正
        self.api_key = api_key
        self.model_name = model_name

    def _call(self, prompt: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        r = requests.post("https://api.deepseek.com/v1/chat/completions",
                        json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"

# 1. PDF／TXT ドキュメントロード＆分割
def load_and_split(folder: str):
    logger_instance = get_logger()
    
    # 必要なディレクトリを作成
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    # 入力ディレクトリの存在確認
    if not os.path.exists(folder):
        raise FileNotFoundError(f"入力ディレクトリ '{folder}' が見つかりません。")
    
    all_docs = []
    
    # --- ファイルロード処理の計測（追加） ---
    # ロガーがあれば計測コンテキストを作成、なければ何もしないコンテキスト
    ctx_load = logger_instance.log_event("ファイル読み込み") if logger_instance else contextmanager(lambda: (yield))()
    
    with ctx_load:
        # 各ファイルタイプの存在確認
        has_pdf = any(f.endswith('.pdf') for f in os.listdir(folder))
        has_txt = any(f.endswith('.txt') for f in os.listdir(folder))
        has_jsonl = any(f.endswith('.jsonl') for f in os.listdir(folder))
        
        if not (has_pdf or has_txt or has_jsonl):
            raise ValueError(f"ディレクトリ '{folder}' に対応するファイル（.pdf, .txt, .jsonl）が見つかりません。")
        
        # PDF をロード
        if has_pdf:
            pdf_loader = DirectoryLoader(
                folder,
                glob="*.pdf",
                loader_cls=PyPDFLoader
            )
            all_docs.extend(pdf_loader.load())
            print(f"[DEBUG] Loaded PDF docs: {len(all_docs)}") # 簡易デバッグ用
        
        # TXT をロード
        if has_txt:
            txt_loader = DirectoryLoader(folder, glob="*.txt")
            all_docs.extend(txt_loader.load())
            print(f"[DEBUG] Loaded TXT docs (accumulated): {len(all_docs)}")
        
        # JSONL をロード
        if has_jsonl:
            jsonl_loader = DirectoryLoader(
                folder,
                glob="*.jsonl",
                loader_cls=JSONLoader,
                loader_kwargs={
                    "jq_schema": "{text: (\"作成日時: \" + .created_at + \"\n\" + .full_text)}",  # 日付情報とツイート本文を抽出
                    "json_lines": True,         # JSONL モードを有効化
                    "text_content": True,       # 抽出結果を文字列として扱う
                    "content_key": "text"       # 本文のキー
                }
            )
            all_docs.extend(jsonl_loader.load())
            print(f"[DEBUG] Loaded JSONL docs (accumulated): {len(all_docs)}")

    if not all_docs:
        raise ValueError("有効なドキュメントが見つかりませんでした。")
    
    print(f"[DEBUG] Total raw docs: {len(all_docs)}")

    # --- 分割処理の計測（追加） ---
    ctx_split = logger_instance.log_event("チャンク分割") if logger_instance else contextmanager(lambda: (yield))()
    
    with ctx_split:
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_docs = splitter.split_documents(all_docs)
        
    print(f"[DEBUG] Total split docs: {len(split_docs)}")
    return split_docs


# 3. 埋め込みモデル初期化
# GPUが利用可能かどうかを確認
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
embeddings = MeasuredEmbeddings(embeddings)

# 4. ChromaDB にベクトル登録
def build_vector_store(docs, embeddings):
    print("ベクトルストアの構築を開始します...")
    logger_instance = get_logger()

    try:
        client = PersistentClient(path=".chroma")
        collection_name = "my_collection"
        
        # コレクションの存在確認と取得
        try:
            # パラメータ指定
            col = client.get_collection(collection_name)
            col = client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",       # 距離計算手法 (l2, cosine, ip)
                    "hnsw:M": 16,                 # デフォルトは16
                    "hnsw:ef_construction": 100,  # デフォルトは100
                    "hnsw:batch_size": 100,       # インデックス更新のバッチサイズ
                    "hnsw:sync_threshold": 1000   # 永続化のしきい値
                }
            )
            existing = col.count()
            print(f"既存コレクション '{collection_name}' を使用 (ドキュメント数: {existing})")
            return col
        except Exception as e:
            print(f"新しいコレクション '{collection_name}' を作成します")
            # 既存のコレクションを削除する処理は削除しました（手動で行うため）
            
            # 新しいコレクションを作成
            col = client.create_collection(collection_name)
            contents = [d.page_content for d in docs]
            
            print("ドキュメントの埋め込みを開始します...")
            # バッチサイズを設定（ChromaDBの制限に合わせる）
            BATCH_SIZE = 5000
            total_docs = len(contents)
            
            for i in range(0, total_docs, BATCH_SIZE):
                batch_contents = contents[i:i + BATCH_SIZE]
                print(f"[INFO] Processing batch {i//BATCH_SIZE + 1}/{(total_docs + BATCH_SIZE - 1)//BATCH_SIZE}")
                
                # バッチごとに埋め込みを計算 (MeasuredEmbeddingsで計測)
                embs = embeddings.embed_documents(batch_contents)
                
                # バッチごとにIDを生成
                batch_ids = [f"doc_{i + j}" for j in range(len(batch_contents))]
                
                # バッチをベクトルストアに追加
                if logger_instance:
                    with logger_instance.log_event(f"ChromaDB登録 (batch {len(batch_contents)})"):
                        col.add(documents=batch_contents, embeddings=embs, ids=batch_ids)
                else:
                    col.add(documents=batch_contents, embeddings=embs, ids=batch_ids)
                
                print(f"[INFO] Added batch of {len(batch_contents)} documents")
            
            print("ベクトルストアへの登録完了")
            new_count = col.count()
            print(f"登録後のドキュメント数: {new_count}")
            return col
            
    except Exception as e:
        print(f"ベクトルストア構築エラー: {e}")
        raise


# 5. RAG チェーン初期化
def init_qa_chain(collection_name: str, deepllm):
    print("QAチェーンを初期化します...")
    try:
        vectorstore = Chroma(
            collection_name="my_collection",
            persist_directory=".chroma",
            embedding_function=embeddings
        )
        # ChromaDBの基本的な設定
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 10  # 取得するドキュメント数
            }
        )

        # 事前指示の設定
        prompt_template = PromptTemplate(
            template="""以下の指示に従って質問に答えてください：

指示：
- 簡潔で分かりやすい文章を使用してください
- 絵文字は使用しないでください
- 日付に関する質問の場合は、該当する期間の情報のみを回答してください
- 該当する期間の情報がない場合は、その旨を明確に伝えてください

context：
{context}

question: {question}

answer:""",
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=deepllm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={
                "verbose": True,
                "prompt": prompt_template
            }
        )
    except Exception as e:
        print(f"QAチェーン初期化エラー: {e}")
        raise

# 6. 全体フロー
if __name__ == "__main__":
    logger = TimelineLogger()
    set_logger(logger)
    load_dotenv()  # .env の内容を os.environ に反映


    # ---------------------------------------------------------
    # モンキーパッチ適用開始 (LangChain対応版)
    # ---------------------------------------------------------
    import chromadb.api.models.Collection

    # 1. ターゲットを Collection クラスに変更
    target_class = chromadb.api.models.Collection.Collection
    original_query_method = target_class.query

    # 2. 差し替え用のメソッドを定義
    def patched_query(self, *args, **kwargs):
        logger_instance = get_logger()
        # ロガー取得成功時のみ計測
        if logger_instance:
            with logger_instance.log_event("ChromaDB検索(パッチ)"):
                return original_query_method(self, *args, **kwargs)
        else:
            return original_query_method(self, *args, **kwargs)

    # 3. メソッドを上書き
    target_class.query = patched_query
    print(f"[INFO] {target_class.__name__}.query に計測パッチを適用しました")
    # ---------------------------------------------------------import time

    # 既存のembed_queryメソッドを退避
    original_embed_query = embeddings.embed_query

    def measured_embed_query(text):
        start = time.time()
        # 本来の処理を実行
        result = original_embed_query(text)
        end = time.time()
        
        duration = end - start
        print(f"[クエリ埋め込み] 完了 ({duration:.4f}秒)")
        # 必要ならここでリストなどに時間を記録する
        return result

    # メソッドを差し替え
    embeddings.embed_query = measured_embed_query




    # 環境変数からキー取得
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

    # ChromaDBクライアントの初期化
    client = PersistentClient(path=".chroma")
    collection_name = "my_collection"
    print("全体フローを開始...")
    with logger.log_event("全体フロー"):
        
        # 2.ドキュメント準備
        with logger.log_event("ドキュメントロードと分割"):
            # コレクションの存在確認
            try:
                collection = client.get_collection(collection_name)
                print(f"既存のコレクション '{collection_name}' を使用します")
            except Exception as e:
                print(f"新しいコレクション '{collection_name}' を作成します")
                # ドキュメント準備
                docs = load_and_split("data/input")
                # ベクトルストア構築
                collection = build_vector_store(docs, embeddings)

        # DeepSeek LLM
        deepllm = DeepSeekAPI(api_key=DEEPSEEK_KEY)

        # QA チェーン
        qa = init_qa_chain(collection_name, deepllm)

        print("チャットを開始します。終了するには 'quit' または 'exit' と入力してください。")
        
        while True:
            # ユーザーからの入力を取得
            # query = input("\n質問を入力してください: ").strip()
            query =  "重要な点を要約して教えて"

            # 終了条件のチェック
            if query.lower() in ['quit', 'exit']:
                print("チャットを終了します。")
                break
                
            # 空の入力をスキップ
            if not query:
                continue
                
            
            # 質問に対する回答を取得
            result = qa.invoke({"query": query}, config={"callbacks": [PerformanceCallbackHandler()]})
            answer = result["result"]
            source_docs = result.get("source_documents", [])
            
            print("\n回答:", answer)
            
            # クエリと回答をMarkdownファイルとして保存
            output_dir = "data/output"
            os.makedirs(output_dir, exist_ok=True)
            
            # タイムスタンプを含むファイル名を生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"qa_{timestamp}.md")
            
            # Markdown形式でクエリと回答を記録
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("# 質問と回答\n\n")
                f.write(f"## 質問\n{query}\n\n")
                f.write(f"## 回答\n{answer}\n\n")
                f.write("## 参照したソースドキュメント\n\n")
                for i, doc in enumerate(source_docs, 1):
                    f.write(f"### ソース {i}\n")
                    f.write(f"```\n{doc.page_content}\n```\n\n")
            
            print(f"\n質問と回答を {output_file} に保存しました")
            break # 1回で終了

    # 3. CSVファイルから、Python(L1)とRust(L2-L4)全てのデータを読み込む
    all_records = logger.load_merged_records()
    
    # 時間順にソート（念のため）
    all_records.sort(key=lambda x: x[1])

    # 4. 相対時刻に変換
    logged_data_for_plot = convert_records_to_relative(all_records)

    # 5. 描画
    plot_timeline(logged_data_for_plot, output_filename="timeline_full_stack.png")
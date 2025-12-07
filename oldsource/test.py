# 必要ライブラリのインポート
# pip install langchain langchain-community langchain-chroma chromadb requests python-dotenv pypdf sentence-transformers
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
import os
import requests
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from dotenv import load_dotenv
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# タイムロギング
class TimelineLogger:
    def __init__(self):
        # (イベント名, 開始時刻, 終了時刻) のタプルを保存するリスト
        self.records = []
        self._start_times = {}

    @contextmanager
    def log_event(self, name: str):
        """'with'文でイベントの開始・終了を記録する"""
        print(f"[{name}] 開始")
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        self.records.append((name, start_time, end_time))
        print(f"[{name}] 完了 ({end_time - start_time:.4f}秒)")


# 1. DeepSeek API 用カスタム LLM
class DeepSeekAPI(LLM):
    api_key: str
    model_name: str

    def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
        super().__init__(api_key=api_key, model_name=model_name)  # ここを修正
        self.api_key = api_key
        self.model_name = model_name

    def _call(self, prompt: str, **kwargs) -> str:
        print("LLM問い合わせの所要時間を計測...")
        llm_start_time = time.time()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model_name, "prompt": prompt}
        r = requests.post("https://api.deepseek.com/v1/chat/completions",
                          json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"

# 1. PDF／TXT ドキュメントロード＆分割
def load_and_split(folder: str):

    # PDF をロード
    pdf_loader = DirectoryLoader(
        folder,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    pdf_docs = pdf_loader.load()
    # TXT も併せてロードする場合
    txt_loader = DirectoryLoader(folder, glob="*.txt")
    txt_docs = txt_loader.load()

    docs = pdf_docs + txt_docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# 3. 埋め込みモデル初期化
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 4. ChromaDB にベクトル登録
def embed_docs(docs, embeddings_model):
    """ドキュメントをベクトルに変換する"""
    
    contents = [d.page_content for d in docs]
    embs = embeddings_model.embed_documents(contents)
    print(f"{len(contents)}件のドキュメントの埋め込みが完了しました。")
    return contents, embs

def build_or_get_chroma_collection(contents, embs):
    """ChromaDBのコレクションを構築または取得する"""
    client = chromadb.PersistentClient(path=".chroma")
    
    try:
        # 既存のコレクションを取得できれば、それを返す
        collection = client.get_collection("my_collection")
        print("既存のコレクションを使用します。")
        return collection
    except Exception:
        # 存在しない場合は新規作成してデータを登録
        print("新しいコレクションを作成し、データを登録します。")
        collection = client.create_collection("my_collection")
        ids = [f"doc_{i}" for i in range(len(contents))]
        collection.add(documents=contents, embeddings=embs, ids=ids)
        return collection

# 5. RAG チェーン初期化
def init_qa_chain(collection, deepllm):
    print("QAチェーンを初期化します...")
    try:
        # Step 1: embedding_function を渡さずに Chroma を初期化
        # これにより、chromadb 内部への不要な引数伝播を回避する
        vectorstore = Chroma(
            client=collection._client,
            collection_name=collection.name,
        )
        
        # Step 2: LangChain のレトリバーが検索に使えるように、
        # Chroma インスタンスの属性に埋め込み関数を手動で設定
        vectorstore.embedding_function = embeddings.embed_query
        
        retriever = vectorstore.as_retriever()
        return RetrievalQA(llm=deepllm, retriever=retriever)
    except Exception as e:
        print(f"QAチェーン初期化エラー: {e}")
        raise
    
# 5. RAG チェーン初期化
def oldinit_qa_chain(collection, deepllm):
    print("QAチェーンを初期化します...")
    try:
        # ChromaDBのコレクションをLangChainのRetrieverに変換
        vectorstore = Chroma(
            client=collection._client,
            collection_name=collection.name,
            embedding_function=embeddings.embed_query
        )
        retriever = vectorstore.as_retriever()
        return RetrievalQA(llm=deepllm, retriever=retriever)
    except Exception as e:
        print(f"QAチェーン初期化エラー: {e}")
        raise
def plot_timeline(records: list, output_filename: str = "timeline.png"):
    """
    (イベント名, 開始時刻, 終了時刻) のリストから
    タイムライン画像を生成・保存する
    """
    if not records:
        print("描画するデータがありません。")
        return

    # 日本語フォントの設定（macOS/Windowsの一般的な例）
    try:
        # for mac
        plt.rcParams['font.family'] = 'Hiragino Sans'
    except:
        try:
            # for windows
            plt.rcParams['font.family'] = 'Meiryo'
        except:
            # for linux/colab etc.
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            print("日本語フォントが見つからないため、デフォルトフォントで描画します。")

    # データから描画用の要素を抽出
    event_names = [rec[0] for rec in records]
    start_times = [rec[1] for rec in records]
    end_times = [rec[2] for rec in records]
    
    # 各イベントの処理時間（棒の長さ）を計算
    durations = [end - start for start, end in zip(start_times, end_times)]

    # 描画処理
    fig, ax = plt.subplots(figsize=(10, len(event_names) * 0.5))

    # 横棒グラフを描画 (left=start_times がタイムラインの鍵)
    ax.barh(event_names, durations, left=start_times, height=0.6)

    # 見た目を整える
    ax.invert_yaxis()  # 最初のイベントが上に来るようにする
    ax.set_xlabel("時間 (秒)")
    ax.set_title("処理タイムライン")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

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


# 6. 全体フロー
if __name__ == "__main__":
    from shared_logger import set_logger
    logger = TimelineLogger()
    set_logger(logger)
    
    load_dotenv()  # .env の内容を os.environ に反映
    print(os.getenv("DEEPSEEK_API_KEY"))
    # 環境変数からキー取得
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
    client = chromadb.PersistentClient(path=".chroma")
    collection = None

    print("全体フローを開始...")
    with logger.log_event("全体フロー"):

        # 2.ドキュメント準備
        with logger.log_event("ドキュメントロードと分割"):
            docs = load_and_split("data/documents")

        try:
            # 既存コレクションの取得を試みる
            collection = client.get_collection("my_collection")
            print("既存のベクトルストアを発見")
        except Exception:
            print("ベクトルストアを新規作成")
            # 3.ベクトル埋め込み
            with logger.log_event("ベクトル埋め込み"):
                contents, embs = embed_docs(docs, embeddings)

            # 4.ベクトルストア構築
            with logger.log_event("`ベクトルストア構築`"):
                collection = build_or_get_chroma_collection(contents, embs)
        

        # DeepSeek LLM と QAチェーンの準備
        deepllm = DeepSeekAPI(api_key=DEEPSEEK_KEY)
        qa = init_qa_chain(collection, deepllm)

        query = "線形代数初学者に向けて, 固有値分解と特異値分解の差異について教えてください"
        
        # Retrieverによる検索処理を計測
        retriever = qa.retriever
        retrieved_docs = []

        # 5. クエリ埋め込み時間
        # 6. ベクトル検索時間
        with logger.log_event("検索全体（埋め込み＋検索）"):
            retrieved_docs = retriever.get_relevant_documents(query)
            print(f"関連ドキュメントを{len(retrieved_docs)}件取得しました。")

        #    取得したドキュメントをコンテキストとしてプロンプトを作成
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"""以下のコンテキスト情報のみを使用して、質問に回答してください。
---
コンテキスト:
{context}
---
質問: {query}
"""
        
        # 7. LLM回答生成
        with logger.log_event("LLM回答生成"):
            answer = deepllm(prompt)

        print("回答:", answer)
        

    print("全体フロー完了")
    
# 3. 記録済みの絶対時刻データを、描画用の相対時刻データに変換 ➡️
logged_data_for_plot = convert_records_to_relative(logger.records)

# 4. 変換したデータをプロット関数に渡す
plot_timeline(logged_data_for_plot)
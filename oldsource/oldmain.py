from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA # 非推奨
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import torch
import dateutil.parser

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
    # 必要なディレクトリを作成
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    # 入力ディレクトリの存在確認
    if not os.path.exists(folder):
        raise FileNotFoundError(f"入力ディレクトリ '{folder}' が見つかりません。")
    
    # 各ファイルタイプの存在確認
    has_pdf = any(f.endswith('.pdf') for f in os.listdir(folder))
    has_txt = any(f.endswith('.txt') for f in os.listdir(folder))
    has_jsonl = any(f.endswith('.jsonl') for f in os.listdir(folder))
    
    if not (has_pdf or has_txt or has_jsonl):
        raise ValueError(f"ディレクトリ '{folder}' に対応するファイル（.pdf, .txt, .jsonl）が見つかりません。")
    
    # PDF をロード
    pdf_docs = []
    if has_pdf:
        pdf_loader = DirectoryLoader(
            folder,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        pdf_docs = pdf_loader.load()
        print(f"[DEBUG] Loaded PDF docs: {len(pdf_docs)}")
    
    # TXT をロード
    txt_docs = []
    if has_txt:
        txt_loader = DirectoryLoader(folder, glob="*.txt")
        txt_docs = txt_loader.load()
        print(f"[DEBUG] Loaded TXT docs: {len(txt_docs)}")
    
    # JSONL をロード
    jsonl_docs = []
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
        jsonl_docs = jsonl_loader.load()
        print(f"[DEBUG] Loaded JSONL docs: {len(jsonl_docs)}")

    all_docs = pdf_docs + txt_docs + jsonl_docs
    if not all_docs:
        raise ValueError("有効なドキュメントが見つかりませんでした。")
    
    print(f"[DEBUG] Total raw docs: {len(all_docs)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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

# 4. ChromaDB にベクトル登録
def build_vector_store(docs, embeddings):
    print("ベクトルストアの構築を開始します...")
    try:
        client = PersistentClient(path=".chroma")
        collection_name = "my_collection"
        
        # コレクションの存在確認と取得
        try:
            col = client.get_collection(collection_name)
            existing = col.count()
            print(f"既存コレクション '{collection_name}' を使用 (ドキュメント数: {existing})")
            return col
        except Exception as e:
            print(f"新しいコレクション '{collection_name}' を作成します")
            # 既存のコレクションを削除（存在する場合）
            try:
                client.delete_collection(collection_name)
                print(f"既存のコレクション '{collection_name}' を削除しました")
            except:
                pass
            
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
                
                # バッチごとに埋め込みを計算
                embs = embeddings.embed_documents(batch_contents)
                
                # バッチごとにIDを生成
                batch_ids = [f"doc_{i + j}" for j in range(len(batch_contents))]
                
                # バッチをベクトルストアに追加
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
    load_dotenv()  # .env の内容を os.environ に反映

    # 環境変数からキー取得
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

    # ChromaDBクライアントの初期化
    client = PersistentClient(path=".chroma")
    collection_name = "my_collection"

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
        query = input("\n質問を入力してください: ").strip()
        
        # 終了条件のチェック
        if query.lower() in ['quit', 'exit']:
            print("チャットを終了します。")
            break
            
        # 空の入力をスキップ
        if not query:
            continue
            
        # 質問に対する回答を取得
        result = qa.invoke({"query": query})
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


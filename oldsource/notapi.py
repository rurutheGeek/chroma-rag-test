from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient
import os
from datetime import datetime
import torch
import dateutil.parser

# 日付文字列をパースする関数（必要に応じて使用）
def parse_twitter_date(date_str):
    try:
        return dateutil.parser.parse(date_str)
    except:
        return None

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
                "jq_schema": "{text: (\"作成日時: \" + .created_at + \"\n\" + .full_text)}",
                "json_lines": True,
                "text_content": True,
                "content_key": "text"
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

# 2. 埋め込みモデル初期化
# GPUが利用可能かどうかを確認
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

# 3. ChromaDB にベクトル登録
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
            try:
                client.delete_collection(collection_name)
                print(f"既存のコレクション '{collection_name}' を削除しました")
            except:
                pass
            
            col = client.create_collection(collection_name)
            contents = [d.page_content for d in docs]
            
            print("ドキュメントの埋め込みを開始します...")
            BATCH_SIZE = 5000
            total_docs = len(contents)
            
            for i in range(0, total_docs, BATCH_SIZE):
                batch_contents = contents[i:i + BATCH_SIZE]
                print(f"[INFO] Processing batch {i//BATCH_SIZE + 1}/{(total_docs + BATCH_SIZE - 1)//BATCH_SIZE}")
                
                embs = embeddings.embed_documents(batch_contents)
                batch_ids = [f"doc_{i + j}" for j in range(len(batch_contents))]
                
                col.add(documents=batch_contents, embeddings=embs, ids=batch_ids)
                print(f"[INFO] Added batch of {len(batch_contents)} documents")
            
            print("ベクトルストアへの登録完了")
            new_count = col.count()
            print(f"登録後のドキュメント数: {new_count}")
            return col
            
    except Exception as e:
        print(f"ベクトルストア構築エラー: {e}")
        raise

# 4. プロンプト生成ロジック (RAG検索のみ実行)
def generate_rag_prompt(query: str, collection_name: str, embedding_function):
    # ChromaDBに接続
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=".chroma",
        embedding_function=embedding_function
    )
    
    # 関連ドキュメントの検索 (Top 10)
    print("関連情報を検索中...")
    results = vectorstore.similarity_search(query, k=10)
    
    # コンテキストの結合
    context_text = "\n\n".join([doc.page_content for doc in results])
    
    # プロンプトテンプレートの作成
    prompt_template = """以下の指示に従って質問に答えてください：

指示：
- 簡潔で分かりやすい文章を使用してください
- 絵文字は使用しないでください
- 日付に関する質問の場合は、該当する期間の情報のみを回答してください
- 該当する期間の情報がない場合は、その旨を明確に伝えてください

context：
{context}

question: {question}

answer:"""

    # テンプレートに値を埋め込む
    final_prompt = prompt_template.format(context=context_text, question=query)
    
    return final_prompt, results

# 5. 全体フロー
if __name__ == "__main__":

    # ChromaDBクライアントの初期化
    client = PersistentClient(path=".chroma")
    collection_name = "my_collection"

    # コレクションの存在確認とデータの準備
    try:
        collection = client.get_collection(collection_name)
        print(f"既存のコレクション '{collection_name}' を使用します")
    except Exception as e:
        print(f"新しいコレクション '{collection_name}' を作成します")
        docs = load_and_split("data/input")
        collection = build_vector_store(docs, embeddings)

    print("チャット（プロンプト生成モード）を開始します。終了するには 'quit' または 'exit' と入力してください。")
    
    while True:
        # ユーザーからの入力を取得
        query = input("\n質問を入力してください: ").strip()
        
        # 終了条件のチェック
        if query.lower() in ['quit', 'exit']:
            print("終了します。")
            break
            
        # 空の入力をスキップ
        if not query:
            continue
            
        # プロンプトの生成
        final_prompt, source_docs = generate_rag_prompt(query, collection_name, embeddings)
        
        print("\n" + "="*50)
        print("生成されたプロンプト:")
        print("="*50)
        print(final_prompt)
        print("="*50)
        
        # プロンプトをMarkdownファイルとして保存
        output_dir = "data/output"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"prompt_{timestamp}.md")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# RAGプロンプト出力\n\n")
            f.write("このファイルは、ユーザーの質問と検索された関連情報を結合したプロンプトです。\n\n")
            f.write("```text\n")
            f.write(final_prompt)
            f.write("\n```\n\n")
            f.write("## 参照したソースドキュメント詳細\n\n")
            for i, doc in enumerate(source_docs, 1):
                f.write(f"### ソース {i}\n")
                f.write(f"```\n{doc.page_content}\n```\n\n")
        
        print(f"\nプロンプトを {output_file} に保存しました")
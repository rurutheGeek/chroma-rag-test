# check_chroma_loc.py
import chromadb
from chromadb import PersistentClient
import inspect

print("--- ChromaDB 内部構造調査 ---")

# 1. あなたのコードと同じ方法でクライアントを作成
client = PersistentClient(path=".chroma")

# 2. 内部APIの実体を取得 (_api が実務を行っているオブジェクトです)
# ※ バージョンによっては構造が違う可能性があるため、段階的にチェックします
if hasattr(client, "_api"):
    internal_api = client._api
    print(f"✅ client._api が見つかりました: {type(internal_api)}")
else:
    print("⚠️ client._api が見つかりません。client自体を調査します。")
    internal_api = client

# 3. 定義されているファイルパスとクラス名を表示
print("\n[特定結果]")
print(f"クラス名: {internal_api.__class__.__name__}")
print(f"ファイル: {inspect.getfile(internal_api.__class__)}")

# 4. queryメソッドの場所を確認
if hasattr(internal_api, "_query"):
    target_method = internal_api._query
    method_name = "_query"
elif hasattr(internal_api, "query"):
    target_method = internal_api.query
    method_name = "query"
else:
    target_method = None
    print("❌ query メソッドが見つかりませんでした。")

if target_method:
    print(f"ターゲットメソッド: {method_name}")
    print(f"メソッド定義行: {inspect.getsourcelines(target_method)[1]} 行目付近")
    
    # パッチに必要なインポートパスを生成して表示
    module_path = internal_api.__class__.__module__
    class_name = internal_api.__class__.__name__
    
    print("\n--- モンキーパッチ用コード ---")
    print(f"import {module_path}")
    print(f"target_class = {module_path}.{class_name}")
    print(f"original_method = target_class.{method_name}")
    print("----------------------------")
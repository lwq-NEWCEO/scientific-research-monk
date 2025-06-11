# create_vector_db.py

import os
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm
import torch # 需要导入 torch 来检查 cuda 可用性

# !!! 请将 DATA_PATH 修改为您知识库文档所在的实际目录 !!!
DATA_PATH = r"E:\DesignThinking\knowledge"

# !!! 请将 DB_FAISS_PATH 修改为您希望保存向量数据库的实际目录 !!!
# 脚本会在这个路径下创建一个名为 'db_faiss' 的子文件夹
DB_FAISS_PATH = r"E:\DesignThinking\knowledge\vectorstore\db_faiss"

# Embedding 模型名称
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 是一个常用的多语言模型
# 可以根据需要更换为其他 Sentence-Transformer 模型
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def create_vector_db():
    """
    加载文档，切分文本，生成 embeddings 并存储到 FAISS 向量数据库中。
    """
    print(f"扫描知识库目录: {DATA_PATH}")
    document_paths = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            document_paths.append(file_path)

    if not document_paths:
        print(f"警告：在目录 '{DATA_PATH}' 中未找到任何文件。请确认路径是否正确且包含文档。")
        return # 如果没有找到文件，直接退出

    documents = []
    print("开始加载文档...")
    for doc_path in tqdm(document_paths, desc="加载文档"):
        try:
            loader = None
            if doc_path.lower().endswith(".txt"):
                loader = TextLoader(doc_path, encoding="utf-8")
            elif doc_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(doc_path)
            elif doc_path.lower().endswith((".docx", ".doc")):
                try:
                    # 优先尝试 Docx2txtLoader (如果安装了 docx2txt)
                    from langchain.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(doc_path)
                except ImportError:
                    # 如果 docx2txt 未安装，回退到 UnstructuredFileLoader
                    print(f"docx2txt 未安装或加载失败，尝试使用 UnstructuredFileLoader 加载 {doc_path}")
                    loader = UnstructuredFileLoader(doc_path)
            else:
                # 对于其他文件类型，尝试使用 UnstructuredFileLoader
                loader = UnstructuredFileLoader(doc_path)

            if loader:
                documents.extend(loader.load())
            else:
                 print(f"跳过不支持的文件类型: {doc_path}")

        except Exception as e:
            print(f"\n加载文档 {doc_path} 失败: {e}")
            print("请检查文件格式或安装相应的库 (如 unstructured 的依赖, Poppler for PDF)。")


    print(f"总共加载的文档数量: {len(documents)}")

    if not documents:
        print("没有成功加载任何文档，无法创建向量数据库。请检查文档或加载器问题。")
        return

    print("开始切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print(f"分割后的文本块数量: {len(texts)}")

    # 过滤掉空文本块
    texts = [text for text in texts if text.page_content.strip()]
    print(f"过滤空文本块后剩余数量: {len(texts)}")

    if not texts:
        print("文本切分后为空，无法创建向量数据库。请检查文档内容或切分参数。")
        return


    # # 打印文本块内容 (可选，用于调试)
    # print("\n--- 部分文本块预览 ---")
    # for i, text in enumerate(texts[:5]): # 只打印前5个
    #      print(f"文本块 {i}: {text.page_content[:200]}...") # 只打印每个文本块的前200个字符
    # print("---------------------\n")


    print(f"开始生成 embeddings，使用模型: {EMBEDDING_MODEL_NAME}")
    # 检查是否有 CUDA 可用，并设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Embedding 模型将在设备 '{device}' 上运行。")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device} # 指定运行设备
        )
        print("Embedding 模型加载成功！")
    except Exception as e:
        print(f"\n加载 Embedding 模型 '{EMBEDDING_MODEL_NAME}' 失败: {e}")
        print("请检查网络连接以下载模型，或确认模型名称是否正确。")
        print("如果使用 CUDA 报错，请检查 PyTorch 和 CUDA 安装配置。")
        return # 加载 embedding 模型失败则退出


    print("开始创建和保存 FAISS 向量数据库...")
    try:
        # 确保保存目录存在
        os.makedirs(DB_FAISS_PATH, exist_ok=True)

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        print(f"向量数据库创建完成，保存到: {DB_FAISS_PATH}")

    except Exception as e:
        print(f"\n创建或保存 FAISS 向量数据库失败: {e}")
        print(f"请检查是否有权限写入目录 '{os.path.dirname(DB_FAISS_PATH)}'，或磁盘空间是否足够。")


if __name__ == "__main__":
    create_vector_db()
    print("\n脚本执行完毕。")

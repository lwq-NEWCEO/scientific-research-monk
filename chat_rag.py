import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import warnings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline  # 已经在顶部导入
warnings.filterwarnings("ignore", category=UserWarning, module='langchain')
warnings.filterwarnings("ignore", category=UserWarning, module='langchain_community') # 如果您使用了 langchain_community

# !!! 请将 MODEL_PATH 修改为您本地大型语言模型文件所在的实际目录 !!!
MODEL_PATH = r"E:\DesignThinking\model\7B"

# !!! 请将 DB_FAISS_PATH 修改为您希望保存向量数据库的实际目录 !!!
DB_FAISS_PATH = r"E:\DesignThinking\knowledge\vectorstore\db_faiss" # 确保这个路径和 create_vector_db.py 中保存的一致

# Embedding 模型名称 (必须与 create_vector_db.py 中使用的模型名称完全一致)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_model():
    """加载模型和 tokenizer"""
    print(f"正在加载模型: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # 对于 14B 模型，device_map='auto' 和 torch_dtype=torch.float16 非常重要以节省显存
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
        print("模型加载成功！")
        return model, tokenizer
    except Exception as e:
        print(f"\n加载模型失败: {e}")
        print(f"请检查模型路径 '{MODEL_PATH}' 是否正确，文件是否完整，以及硬件（特别是显存）是否足够加载模型（即使是 float16）。")
        # 在加载模型失败时退出，因为后续无法进行
        exit()


def load_vector_db():
    """加载 FAISS 向量数据库"""
    print(f"正在加载向量数据库: {DB_FAISS_PATH}")
    try:
        # Embedding 模型必须与创建数据库时使用的模型一致
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Embedding 模型将在设备 '{device}' 上运行。") # 加载 FAISS 需要 embedding 模型
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                           model_kwargs={'device': device})

        # 添加 allow_dangerous_deserialization=True 是通常加载本地FAISS数据库所必需的
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("向量数据库加载成功！")
        return db
    except Exception as e:
        print(f"\n加载向量数据库失败: {e}")
        print(f"请检查路径 '{DB_FAISS_PATH}' 是否正确，以及该目录下是否存在 'index.faiss' 和 'index.pkl' 文件。")
        print("如果文件存在，请确认创建数据库时使用的 embedding 模型和当前加载时的模型一致。")
        # 在加载数据库失败时退出
        exit()

def create_rag_chain(llm, db): # 修改函数签名，只接收llm和db
    """创建 RAG 链"""
    print("正在创建 RAG 检索问答链...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # 使用适配 langchain 接口的 llm
        chain_type="stuff",  # "stuff" 是最简单的 chain_type, 适合小文档
        retriever=db.as_retriever(search_kwargs={'k': 3}),  # 从向量数据库中检索最相关的 3 个文档
        return_source_documents=True,  # 返回源文档
        chain_type_kwargs={"prompt": prompt} # 使用自定义 prompt
    )
    print("RAG 链创建成功！")
    return qa_chain

# 自定义 Prompt
from langchain.prompts import PromptTemplate # 确保导入了 PromptTemplate
prompt_template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
上下文：{context}
问题：{question}
有用的回答："""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# 定义 generate_response 函数
def generate_response(qa_chain, query):
    """生成回复"""
    response = qa_chain.invoke({"query": query})
    return response["result"], response["source_documents"]

if __name__ == "__main__":
    # 确保 embedding 模型名称一致性检查 (可选，但推荐)
    print(f"使用的 Embedding 模型名称: {EMBEDDING_MODEL_NAME}")

    # 检查 CUDA 可用性并在加载模型前打印
    if torch.cuda.is_available():
        print(f"检测到 CUDA 可用。将尝试使用 GPU。显卡名称: {torch.cuda.get_device_name(0)}")
    else:
        print("未检测到 CUDA。将使用 CPU 运行模型和 Embedding。")


    model, tokenizer = load_model() # 加载 HF 模型和 tokenizer
    db = load_vector_db() # 加载向量数据库

    print("正在创建 Hugging Face Pipeline...")
    try:
        # 配置 Pipeline 参数
        pipe = pipeline(
            "text-generation",
            model=model,        # 使用加载的模型
            tokenizer=tokenizer,# 使用加载的 tokenizer
            torch_dtype=torch.float16, # 与模型加载时一致，节省显存
            device_map="auto",  # 与模型加载时一致，自动分配设备
            max_new_tokens=512, # 控制生成回答的最大长度，可以调整
            do_sample=True,     # 启用采样生成，增加多样性
            top_p=0.9,          # 采样参数
            temperature=0.7,    # 控制回答的创造性，0.7 是一个常用的值
            num_return_sequences=1 # 生成一个序列
        )
        print("Hugging Face Pipeline 创建成功！")

        # 将 Pipeline 封装成 LangChain LLM 对象
        llm = HuggingFacePipeline(pipeline=pipe)
        print("LangChain LLM 适配器创建成功！")

    except Exception as e:
        print(f"\n创建 Hugging Face Pipeline 或 LLM 适配器失败: {e}")
        print("请检查transformers, torch版本，以及硬件是否支持当前配置。")
        exit()

    # 创建 RAG 链 (现在只需传入 llm 和 db)
    qa_chain = create_rag_chain(llm, db)

    print("\n欢迎使用 DeepSeek R1 Distill Qwen 7B (RAG)！")
    print("输入 'exit' 退出对话。")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("感谢使用，再见！")
            break
        if not user_input.strip(): # 避免空输入
            continue

        try:
            response, source_documents = generate_response(qa_chain, user_input)
            print("\nDeepSeek: " + response)
            print("\n来源文档:")
            if source_documents:
                for i, doc in enumerate(source_documents):

                    source_info = doc.metadata.get('source', '未知来源')

                    print(f"  - {source_info}") # 只打印来源路径
            else:
                 print("  （未能找到相关来源文档）") # 如果 retrieval 没找到文档

        except Exception as e:
            print(f"\n发生错误: {e}")
            print("请检查你的输入或模型/数据库加载是否正常。")




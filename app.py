# app.py
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# 尝试导入更新的 LangChain 集成，如果失败则使用旧版
try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_huggingface.llms import HuggingFacePipeline
    print("Using langchain_huggingface integrations.")
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
    print("Using langchain_community integrations (langchain-huggingface not found or older version).")

# LangChain 相关组件导入
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
# 导入 Document 类型，用于类型提示（可选，但推荐）
from langchain.schema import Document
import warnings
import time
import os
import tempfile
import shutil # 用于清理临时目录

# --- 配置信息 ---
warnings.filterwarnings("ignore", category=UserWarning) # 忽略特定类型的用户警告
MODEL_PATH = r"E:\DesignThinking\model\7B" # LLM 模型路径
DB_FAISS_PATH = r"E:\DesignThinking\knowledge\vectorstore\db_faiss" # 预先构建好的全局 Faiss 向量数据库路径
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Embedding 模型名称
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 判断使用 GPU 还是 CPU
ANSWER_MARKER = "有用的回答：" # 定义用于提取最终答案的标记

# --- 缓存资源加载函数 (保持不变) ---
@st.cache_resource
def load_embedding_model():
    # ... (代码同上) ...
    print(f"缓存未命中：正在加载 Embedding 模型: {EMBEDDING_MODEL_NAME} 到 {DEVICE}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': DEVICE}
        )
        print("Embedding 模型加载成功！")
        return embeddings
    except Exception as e:
        st.error(f"加载 Embedding 模型失败: {e}")
        st.stop()

@st.cache_resource
def load_llm_and_tokenizer():
    # ... (代码同上) ...
    print(f"缓存未命中：正在加载 LLM: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float16,
            device_map="auto", max_new_tokens=512, do_sample=True, top_p=0.9,
            temperature=0.7, num_return_sequences=1
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        print("LLM、Tokenizer、Pipeline 加载成功！")
        return llm
    except Exception as e:
        st.error(f"加载 LLM 或创建 Pipeline 失败: {e}")
        st.warning("请检查模型路径、文件完整性以及 GPU 显存是否足够。")
        st.stop()

@st.cache_resource
def load_and_get_global_db(_embeddings):
    # ... (代码同上) ...
    print(f"缓存未命中：正在检查并加载全局向量数据库: {DB_FAISS_PATH}")
    if not os.path.exists(DB_FAISS_PATH):
         print(f"全局向量数据库路径不存在: {DB_FAISS_PATH}。")
         return None
    try:
        db = FAISS.load_local(DB_FAISS_PATH, _embeddings, allow_dangerous_deserialization=True)
        print("全局向量数据库加载成功！")
        st.session_state.global_db = db
        return db
    except Exception as e:
        st.error(f"加载全局向量数据库失败: {e}")
        return None

# --- 文件处理函数 (保持不变) ---
def process_uploaded_files(uploaded_files, embeddings):
    # ... (代码同上) ...
    documents = []
    temp_dir = tempfile.mkdtemp()
    loaded_files_info = []
    try:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = None
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            try:
                if file_ext == ".txt": loader = TextLoader(file_path, encoding="utf-8")
                elif file_ext == ".pdf": loader = PyPDFLoader(file_path)
                elif file_ext in (".docx", ".doc"):
                    try: loader = Docx2txtLoader(file_path)
                    except Exception: loader = UnstructuredFileLoader(file_path)
                else: loader = UnstructuredFileLoader(file_path)

                if loader:
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata["source"] = f"上传文件: {uploaded_file.name}"
                    documents.extend(loaded_docs)
                    loaded_files_info.append(uploaded_file.name)
            except Exception as load_err:
                st.warning(f"加载或处理文件 {uploaded_file.name} 时跳过: {load_err}")

        if not documents: return None, []
        print(f"从上传的文件加载了 {len(documents)} 个文档片段。")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        texts = [text for text in texts if text.page_content.strip()]
        print(f"分割为 {len(texts)} 个文本块。")
        if not texts: return None, []
        with st.spinner("正在为上传的文件创建向量索引..."):
            uploaded_db = FAISS.from_documents(texts, embeddings)
        print("上传文件的向量索引创建成功！")
        return uploaded_db, loaded_files_info
    finally:
        shutil.rmtree(temp_dir)
        print(f"临时目录 {temp_dir} 已清理。")

# --- 定义 Prompt 模板 (保持不变) ---
# 注意：虽然Prompt要求模型输出"有用的回答："，但模型不一定严格遵守
prompt_template_str = """根据以下上下文信息，简洁和专业地回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许编造答案。在回答末尾说明答案主要基于哪些信息来源。
上下文：
{context}
问题：{question}
有用的回答："""
prompt_template = PromptTemplate(
    template=prompt_template_str, input_variables=["context", "question"]
)

# --- 创建 RAG 链的函数 (保持不变) ---
def create_rag_chain(llm, retriever):
    # ... (代码同上) ...
    print("创建 RAG 链 (使用提供的 retriever)...")
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        print("RAG 链创建成功！")
        return qa_chain
    except Exception as e:
        st.error(f"创建 RAG 链失败: {e}")
        st.stop()

# --- 辅助函数：格式化上下文用于显示 (保持不变) ---
def format_context_for_display(source_docs: list[Document]) -> str:
    # ... (代码同上) ...
    if not source_docs:
        return "没有检索到相关上下文。"
    context_parts = []
    for i, doc in enumerate(source_docs):
        source_name = doc.metadata.get('source', f'未知来源 {i+1}')
        context_parts.append(f"#### 来源: {source_name}\n```\n{doc.page_content}\n```")
    return "\n\n---\n\n".join(context_parts)

# --- Streamlit 应用界面 (大部分不变) ---
st.set_page_config(page_title="混合知识库对话", page_icon=":books:")
st.title("科研僧 RAG 对话 (全局+上传)")
st.caption(f"由 {MODEL_PATH.split(os.sep)[-1]} 和 LangChain 驱动")

# --- 加载核心组件 (保持不变) ---
with st.spinner("正在加载核心组件，请稍候..."):
    embeddings = load_embedding_model()
    llm = load_llm_and_tokenizer()
    load_and_get_global_db(embeddings)

# --- 侧边栏文件上传 (保持不变) ---
st.sidebar.title("上传新文件")
uploaded_files = st.sidebar.file_uploader(
    "选择 TXT, PDF, DOCX 文件",
    type=["txt", "pdf", "docx", "doc"],
    accept_multiple_files=True,
    key="file_uploader"
)
# --- 处理文件上传逻辑 (保持不变) ---
if uploaded_files:
    uploaded_filenames = sorted([f.name for f in uploaded_files])
    if "processed_filenames" not in st.session_state or st.session_state.processed_filenames != uploaded_filenames:
        with st.spinner(f"正在处理 {len(uploaded_files)} 个上传的文件..."):
            st.session_state.uploaded_db, loaded_files_info = process_uploaded_files(uploaded_files, embeddings)
            if st.session_state.uploaded_db:
                st.session_state.loaded_files_info = loaded_files_info
                st.session_state.processed_filenames = uploaded_filenames
                st.sidebar.success(f"成功处理: {', '.join(loaded_files_info)}")
            else:
                if "uploaded_db" in st.session_state: del st.session_state.uploaded_db
                if "loaded_files_info" in st.session_state: del st.session_state.loaded_files_info
                if "processed_filenames" in st.session_state: del st.session_state.processed_filenames
                st.sidebar.warning("未从上传文件中加载有效内容。")
elif not uploaded_files and "processed_filenames" in st.session_state:
     if "uploaded_db" in st.session_state: del st.session_state.uploaded_db
     if "loaded_files_info" in st.session_state: del st.session_state.loaded_files_info
     if "processed_filenames" in st.session_state: del st.session_state.processed_filenames
     st.sidebar.info("已清除上传文件状态。")
     st.rerun()

# --- 构建 Ensemble Retriever (保持不变) ---
retriever_list = []
active_sources = []
if "global_db" in st.session_state and st.session_state.global_db:
    global_retriever = st.session_state.global_db.as_retriever(search_kwargs={'k': 3})
    retriever_list.append(global_retriever)
    active_sources.append("全局知识库")
if "uploaded_db" in st.session_state and st.session_state.uploaded_db:
    uploaded_retriever = st.session_state.uploaded_db.as_retriever(search_kwargs={'k': 3})
    retriever_list.append(uploaded_retriever)
    active_sources.append(f"上传的文件 ({', '.join(st.session_state.loaded_files_info)})")
if not retriever_list:
    st.error("错误：没有可用的知识库。请确保全局数据库存在或上传有效文件。")
    st.stop()
ensemble_retriever = EnsembleRetriever(retrievers=retriever_list)
st.info(f"当前查询范围: {'; '.join(active_sources)}")

# --- 创建 RAG 链 (保持不变) ---
qa_chain = create_rag_chain(llm, ensemble_retriever)

# --- 聊天界面 (初始化和历史记录显示逻辑不变) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是科研僧，有什么我可以帮你的吗？", "sources": [], "context_str": ""}]

# --- 显示历史聊天记录 (逻辑不变) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("context_str"):
             with st.expander("查看思考过程 (检索到的上下文)", expanded=False):
                 st.markdown(message["context_str"], unsafe_allow_html=True)
        st.markdown(message["content"])


# --- 处理新的用户输入 ---
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息 (不变)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": [], "context_str": ""})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 生成并显示助手的回应 ---
    with st.chat_message("assistant"):
        context_expander_placeholder = st.empty() # 用于放置思考过程的展开项
        message_placeholder = st.empty() # 用于放置最终回答

        # --- 初始化变量 ---
        final_display_content = "" # **修改点:** 用于存储最终要在界面显示的内容
        raw_llm_output = ""      # 存储 LLM 的原始输出
        retrieved_sources = []   # 存储检索到的源文档列表
        formatted_context = ""   # 存储格式化后的上下文文本

        with st.spinner("思考中 (查询全局库和上传文件)..."):
            try:
                start_time = time.time()
                response = qa_chain.invoke({"query": prompt}) # 调用 RAG 链
                end_time = time.time()

                raw_llm_output = response["result"] # 获取 LLM 的原始输出
                retrieved_sources = response["source_documents"] # 获取检索到的文档
                print(f"回答生成耗时: {end_time - start_time:.2f} 秒")
                print(f"LLM 原始输出:\n{raw_llm_output}") # 打印原始输出，方便调试

                # --- ***** 关键修改：提取 "有用的回答：" 之后的内容 ***** ---
                marker_index = raw_llm_output.find(ANSWER_MARKER) # 查找标记的位置
                if marker_index != -1:
                    # 如果找到标记，提取标记之后的部分，并去除首尾空格
                    extracted_answer = raw_llm_output[marker_index + len(ANSWER_MARKER):]
                    final_display_content = extracted_answer.strip()
                    print(f"提取到的内容:\n{final_display_content}")
                else:
                    # 如果未找到标记，则将整个原始输出作为最终显示内容（去除首尾空格）
                    final_display_content = raw_llm_output.strip()
                    print(f"未找到标记 '{ANSWER_MARKER}'，使用原始输出。")
                # --- ***** 提取结束 ***** ---

                # --- 格式化 RAG 上下文 (不变) ---
                formatted_context = format_context_for_display(retrieved_sources)

                # --- 可选：附加来源信息到 *提取后的* 最终内容 ---
                if retrieved_sources:
                     unique_source_names = sorted(list(set(doc.metadata.get('source', '未知来源') for doc in retrieved_sources)))
                     source_appendix = "\n\n---\n*信息来源: " + ", ".join(unique_source_names) + "*"
                     final_display_content += source_appendix # 附加到提取后的内容上

            except Exception as e:
                # 错误处理 (不变)
                st.error(f"生成回答时出错: {e}")
                final_display_content = "抱歉，处理您的问题时遇到了错误。"
                formatted_context = "处理问题时发生错误，无法显示上下文。"
                print(f"错误详情: {e}")

        # --- 显示包含 RAG 上下文的展开项 (不变) ---
        with context_expander_placeholder.container():
            with st.expander("查看思考过程 (检索到的上下文)", expanded=False):
                st.markdown(formatted_context, unsafe_allow_html=True)

        # --- 显示最终提取或处理后的内容 (不加粗) ---
        message_placeholder.markdown(final_display_content)

    # --- 将助手的回应（提取后的内容和上下文）添加到 session_state ---
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_display_content, # **修改点:** 存储的是提取后的、最终显示的内容
        "sources": retrieved_sources,     # 存储原始的源文档对象列表
        "context_str": formatted_context  # 存储格式化后的 RAG 上下文文本
    })

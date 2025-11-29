"""RAG模型和向量检索"""
import os
import re
import torch
import faiss
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
from config import EMBEDDING_MODEL_PATH, DATA_FILE_PATH, FAISS_INDEX_PATH


@st.cache_resource
def load_embedding_model():
    """加载嵌入模型用于RAG"""
    try:
        print(f"[INFO] 正在加载嵌入模型 tokenizer: {EMBEDDING_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
        print(f"[INFO] 正在加载嵌入模型: {EMBEDDING_MODEL_PATH}")
        
        # 尝试使用 weights_only=False 兼容 PyTorch 2.6
        import torch
        torch_version = torch.__version__
        print(f"[INFO] PyTorch 版本: {torch_version}")
        
        try:
            model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
        except Exception as e1:
            if "weights_only" in str(e1) or "git-lfs" in str(e1):
                print("[WARNING] 尝试使用 trust_remote_code=True 重新加载...")
                model = AutoModel.from_pretrained(
                    EMBEDDING_MODEL_PATH,
                    trust_remote_code=True,
                    use_safetensors=False
                )
            else:
                raise e1
        
        model.eval()
        print("[SUCCESS] 嵌入模型加载成功！")
        return model, tokenizer
    except Exception as e:
        print(f"[ERROR] 嵌入模型加载失败: {type(e).__name__}")
        print(f"[ERROR] 详细错误: {str(e)}")
        print(f"[ERROR] 模型路径: {EMBEDDING_MODEL_PATH}")
        
        if "git-lfs" in str(e):
            print("\n[解决方案] 模型文件未完整下载，请执行以下命令：")
            print(f"  cd {EMBEDDING_MODEL_PATH}")
            print("  git lfs install")
            print("  git lfs pull")
        
        import traceback
        print(f"[ERROR] 完整堆栈:\n{traceback.format_exc()}")
        return None, None


def load_and_process_data():
    """加载并处理数据文件，构建知识库"""
    try:
        print(f"[INFO] 正在加载知识库数据: {DATA_FILE_PATH}")
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 删除所有 HTML 标签
        soup = BeautifulSoup(content, 'html.parser')
        clean_text = soup.get_text()
        
        # 删除电话号码
        clean_text = re.sub(r'电话[:：]?[^\n]*', '', clean_text)
        
        # 按行分割，每行是一个景点
        lines = clean_text.strip().split('\n')
        knowledge_base = []
        
        for line in lines:
            line = line.strip()
            if line and '——' in line:  # 确保是景点数据行
                # 清理多余空格
                line = ' '.join(line.split())
                knowledge_base.append(line)
        
        print(f"[SUCCESS] 知识库加载成功，共 {len(knowledge_base)} 条数据")
        if len(knowledge_base) > 0:
            print(f"[INFO] 示例数据: {knowledge_base[0][:100]}...")
        return knowledge_base
    except Exception as e:
        print(f"[ERROR] 数据文件加载失败: {type(e).__name__}")
        print(f"[ERROR] 详细错误: {str(e)}")
        print(f"[ERROR] 数据路径: {DATA_FILE_PATH}")
        return []


def encode_text(text, model, tokenizer):
    """将文本编码为向量"""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')
        return embeddings
    except Exception:
        return None


def initialize_rag():
    """初始化RAG系统：加载模型、数据并构建FAISS索引"""
    if st.session_state.embedding_model is None:
        st.session_state.embedding_model, st.session_state.embedding_tokenizer = load_embedding_model()
    
    if st.session_state.embedding_model is None:
        print("[WARNING] 嵌入模型未加载，RAG功能不可用")
        return False
    
    if not st.session_state.knowledge_base:
        st.session_state.knowledge_base = load_and_process_data()
    
    if not st.session_state.knowledge_base:
        print("[WARNING] 知识库为空，RAG功能不可用")
        return False
    
    if st.session_state.faiss_index is None:
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                print(f"[INFO] 从缓存加载 FAISS 索引: {FAISS_INDEX_PATH}")
                st.session_state.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                print("[SUCCESS] FAISS 索引加载成功")
                return True
            except Exception as e:
                print(f"[WARNING] FAISS 索引加载失败，将重新构建: {e}")
        
        try:
            print("[INFO] 开始构建 FAISS 索引...")
            sample_embedding = encode_text(
                st.session_state.knowledge_base[0],
                st.session_state.embedding_model,
                st.session_state.embedding_tokenizer
            )
            
            if sample_embedding is None:
                print("[ERROR] 无法生成样本向量")
                return False
            
            d = sample_embedding.shape[1]
            index = faiss.IndexFlatL2(d)
            print(f"[INFO] 向量维度: {d}")
            
            for i, text in enumerate(st.session_state.knowledge_base):
                embedding = encode_text(
                    text,
                    st.session_state.embedding_model,
                    st.session_state.embedding_tokenizer
                )
                if embedding is not None:
                    index.add(embedding)
                if (i + 1) % 10 == 0:
                    print(f"[INFO] 已处理 {i + 1}/{len(st.session_state.knowledge_base)} 条数据")
            
            st.session_state.faiss_index = index
            print(f"[SUCCESS] FAISS 索引构建完成，共 {index.ntotal} 条向量")
            
            try:
                os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
                faiss.write_index(index, FAISS_INDEX_PATH)
                print(f"[SUCCESS] FAISS 索引已保存到: {FAISS_INDEX_PATH}")
            except Exception as e:
                print(f"[WARNING] FAISS 索引保存失败: {e}")
            
            return True
        except Exception as e:
            print(f"[ERROR] FAISS索引构建失败: {type(e).__name__}")
            print(f"[ERROR] 详细错误: {str(e)}")
            import traceback
            print(f"[ERROR] 完整堆栈:\n{traceback.format_exc()}")
            return False
    
    return True


def retrieve_relevant_context(query, top_k=3):
    """从知识库中检索与查询最相关的上下文"""
    if st.session_state.faiss_index is None or not st.session_state.knowledge_base:
        return ""
    
    try:
        query_embedding = encode_text(
            query,
            st.session_state.embedding_model,
            st.session_state.embedding_tokenizer
        )
        
        if query_embedding is None:
            return ""
        
        D, I = st.session_state.faiss_index.search(query_embedding, top_k)
        
        relevant_texts = []
        for idx in I[0]:
            if 0 <= idx < len(st.session_state.knowledge_base):
                relevant_texts.append(st.session_state.knowledge_base[idx])
        
        if relevant_texts:
            context = "\n\n相关景点信息：\n" + "\n".join([f"- {text}" for text in relevant_texts])
            return context
        
        return ""
    except Exception:
        return ""

"""主应用入口"""
import re
import streamlit as st
from services.recommendation import recommend_destination
from services.chat import generate_response
from ui.sidebar import render_sidebar
from ui.chat import display_chat_history
from models.llm_model import load_model
from models.rag_model import initialize_rag

# 页面配置
st.set_page_config(
    page_title="TourTools LP - 旅游推荐系统",
    page_icon="🌍",
    layout="wide"
)

# 初始化session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'destination' not in st.session_state:
    st.session_state.destination = None
if 'destination_context' not in st.session_state:
    st.session_state.destination_context = ""
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'embedding_tokenizer' not in st.session_state:
    st.session_state.embedding_tokenizer = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []

# 预加载模型（启动时，静默加载）
if st.session_state.model is None or st.session_state.tokenizer is None:
    print("[STARTUP] 开始预加载 LLM 模型...")
    st.session_state.model, st.session_state.tokenizer = load_model()

# 预加载 RAG 系统（启动时，静默加载）
if st.session_state.embedding_model is None:
    print("[STARTUP] 开始初始化 RAG 系统...")
    initialize_rag()

# 主界面
st.title("🌍 TourTools LP - 智能旅游推荐系统")
st.markdown("---")

# 渲染侧边栏
mountain_score, ocean_score, plain_score, user_question = render_sidebar()

# 开始推荐按钮
if st.button("🎯 开始推荐", type="primary", use_container_width=True):
    # 清空历史并重新推荐
    st.session_state.messages = []
    st.session_state.destination = None
    st.session_state.destination_context = ""

    with st.spinner("AI正在根据您的偏好分析最佳目的地..."):
        destination, advice, image_base64, mime_type = recommend_destination(
            mountain_score, ocean_score, plain_score
        )

    # 保存推荐信息
    st.session_state.destination = destination
    
    # 构建目的地上下文
    st.session_state.destination_context = f"""
            当前推荐目的地：{destination}
            用户地貌偏好：山川 {mountain_score}分，海洋 {ocean_score}分，平原 {plain_score}分
            推荐理由：{advice}
    """
    
    # 清理 advice 中的 HTML 标签
    clean_advice = re.sub(r'<[^>]+>', '', advice)
    
    # 添加系统消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"根据您的偏好，我为您推荐：**{destination}**\n\n{clean_advice}",
        "image": image_base64,
        "mime_type": mime_type
    })
    
    # 如果用户有初始问题，自动回答
    if user_question.strip():
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })
        response = generate_response(user_question)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
    
    st.rerun()

# 显示推荐结果和对话历史
if st.session_state.destination:
    st.markdown("---")
    st.subheader(f"🎉 推荐目的地：{st.session_state.destination}") 
    # 显示所有消息
    display_chat_history()
    
    # 对话输入框
    st.markdown("---")
    user_input = st.chat_input(f"问问关于 {st.session_state.destination} 的问题吧...")
    
    if user_input:
        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # 生成AI回复
        with st.spinner("正在思考..."):
            response = generate_response(user_input)
        
        # 添加AI回复
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        st.rerun()

else:
    st.info("👈 请在左侧设置您的地貌偏好，然后点击「开始推荐」按钮获取旅游目的地推荐")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>TourTools LP - 基于 Gemma2 的智能旅游推荐系统</p>
    <p>支持地貌偏好分析、目的地推荐和智能问答</p>
</div>
""", unsafe_allow_html=True)

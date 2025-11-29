"""侧边栏UI组件"""
import streamlit as st


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.header("📊 地貌偏好设置")
        st.markdown("请为以下三种地貌类型打分（0-10分）：")
        
        mountain_score = st.slider("🏔️ 山川", 0, 10, 5, help="您对山川地貌的喜好程度")
        ocean_score = st.slider("🌊 海洋", 0, 10, 5, help="您对海洋地貌的喜好程度")
        plain_score = st.slider("🏙️ 平原", 0, 10, 5, help="您对平原地貌的喜好程度")
        
        st.markdown("---")
        user_question = st.text_area(
            "初始问题（可选）",
            placeholder="例如：这个地方有什么特色美食？",
            key="initial_question",
            help="获取推荐后会自动回答这个问题"
        )
    
    return mountain_score, ocean_score, plain_score, user_question

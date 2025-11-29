"""聊天 UI 组件"""
import base64
import streamlit as st


def render_text(content: str):
    """渲染文本消息"""
    st.markdown(content)


def render_image(image_b64: str, caption: str = "图片"):
    """渲染图片消息"""
    try:
        img_bytes = base64.b64decode(image_b64)
        st.image(img_bytes, caption=caption, use_container_width=True)
    except Exception as e:
        st.error(f"图片显示失败: {e}")


def render_message(msg: dict):
    """渲染单条消息，根据类型自动选择组件"""

    role = msg.get("role", "assistant")

    with st.chat_message(role):
        # 文本部分（如果有）
        if text := msg.get("content"):
            render_text(text)

        # 图片部分（如果有）
        if img := msg.get("image"):
            caption = msg.get("caption") or f"{st.session_state.destination} 景观"
            render_image(img, caption)


def display_chat_history():
    """显示聊天历史"""
    for msg in st.session_state.messages:
        render_message(msg)

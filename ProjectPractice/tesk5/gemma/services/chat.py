"""聊天服务"""
import streamlit as st
from models.llm_model import load_model, load_prompt_template
from models.rag_model import initialize_rag, retrieve_relevant_context


def generate_response(user_input):
    """
    根据用户输入生成AI回复（集成RAG功能）
    """
    # 加载模型和tokenizer
    if st.session_state.model is None or st.session_state.tokenizer is None:
        st.session_state.model, st.session_state.tokenizer = load_model()
    
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    
    if model is None or tokenizer is None:
        return "抱歉，模型加载失败，无法回答您的问题。"

    # 初始化RAG系统（如果尚未初始化）
    rag_available = initialize_rag()
    
    # 使用RAG检索相关上下文
    rag_context = ""
    if rag_available:
        rag_context = retrieve_relevant_context(user_input, top_k=3)

    # 从文件加载系统提示词模板
    prompt_template = load_prompt_template()
    
    # 组合所有上下文信息
    combined_context = st.session_state.destination_context
    if rag_context:
        combined_context += f"\n\n{rag_context}"
    
    # 替换上下文信息
    system_prompt = prompt_template.replace("{destination_context}", combined_context)

    # 构建对话历史（只保留最近10条消息）
    recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
    
    conversation = []
    for msg in recent_messages:
        if msg["role"] == "user":
            conversation.append(f"用户: {msg['content']}")
        else:
            conversation.append(f"AI: {msg['content']}")
    
    conversation.append(f"用户: {user_input}")
    
    full_prompt = f"{system_prompt}\n\n" + "\n".join(conversation) + "\nAI:"

    # 编码输入
    input_ids = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # 生成回答
    output = model.generate(
        **input_ids,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # 解码并提取新生成部分
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 提取AI回复部分
    if "AI:" in decoded_output:
        parts = decoded_output.split("AI:")
        response = parts[-1].strip()
    else:
        response = decoded_output[len(full_prompt):].strip()
    
    # 清理可能的重复内容
    lines = response.split('\n')
    if len(lines) > 1:
        unique_lines = []
        for line in lines:
            if line.strip() and line.strip() not in unique_lines:
                unique_lines.append(line.strip())
        response = '\n'.join(unique_lines[:5])
    
    return response

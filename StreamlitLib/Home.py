import pymysql
import streamlit as st
import base64
import os
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. 页面配置 ---
st.set_page_config(page_title="智能旅游问答系统", page_icon="✈️", layout="wide")


# --- 2. 模型加载 (使用缓存避免重复加载) ---
@st.cache_resource
def get_model():
    # 指向 Task 4 中合并后的模型路径
    model_path = "./gemma2_lora_finetune_output/merged_model"
    # 实际部署通常判断是否有GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None, None, None

    model.eval()  # 切换到评估模式
    return tokenizer, model, device


# 加载模型
tokenizer, model, device = get_model()


# --- 3. 数据库连接 ---
def connect_db():
    return pymysql.connect(
        host="localhost",
        user="root",  # 注意文档中是root
        password="tour123",  # 文档中的密码
        database="Tour",
        charset="utf8mb4"
    )


# --- 4. 业务逻辑函数 ---

def get_destination_info(destination_en_name):
    """从数据库获取图片并转为base64"""
    conn = connect_db()
    cursor = conn.cursor()
    # 假设表结构 destination (id, city, pic, ...)
    # 这里的 destination_en_name 对应表里的 city 字段
    query = "SELECT pic FROM destination WHERE city = %s"
    try:
        cursor.execute(query, (destination_en_name,))
        results = cursor.fetchall()  # 获取所有结果
    except Exception as e:
        st.error(f"DB Error: {e}")
        return None
    finally:
        conn.close()

    if results:
        # 结果是元组列表 [('/opt/data/...jpg',), ...]
        image_path = random.choice(results)[0]
        return image_path
    return None


def image_file_to_base64(image_path):
    """将本地图片转换为Base64编码，以便在Web中显示"""
    if not image_path or not os.path.exists(image_path):
        return None, None

    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode('utf-8')

    ext = image_path.split('.')[-1].lower()
    mime_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
    return base64_data, mime_type


def recommend_destination(mountain, ocean, plain):
    """
    基于规则的推荐逻辑（非AI）。
    根据用户滑块的分数，映射到特定城市。
    """
    # 映射表：中文名 -> {英文名, 推荐语, 对应的分数变量}
    rec_map = {
        "山西": {"en": "Shanxi", "advice": "您似乎最喜欢山川地貌...", "score": mountain},
        "海南": {"en": "Hainan", "advice": "您对海洋有浓厚的兴趣...", "score": ocean},
        "北京": {"en": "Beijing", "advice": "您更倾向于平原景观...", "score": plain}
    }

    # 找出得分最高的城市
    # 逻辑：创建一个 {城市名: 分数} 的字典，然后取 max
    scores = {city: info['score'] for city, info in rec_map.items()}
    best_city_cn = max(scores, key=scores.get)

    info = rec_map[best_city_cn]

    # 获取图片
    img_path = get_destination_info(info['en'])
    b64_str, mime = image_file_to_base64(img_path)

    return best_city_cn, info['advice'], b64_str, mime


# --- 5. 大模型推理函数 ---
def predict(user_input, history=[]):
    # 构建 Prompt (类似 RAG 的思路，将推荐结果作为上下文，或者纯聊天)
    # 文档中使用了 prompt_history 拼接历史对话

    # 格式化历史对话
    history_text = "\n".join([f"用户: {q}\nAI: {a}" for q, a in history])

    # 构建完整 Prompt
    full_prompt = (
        f"{history_text}\n"
        f"用户: {user_input}\nAI:"
    ).strip()

    # 编码
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,  # 创造性
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # 解码
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取回答：去掉 prompt 部分，只保留 AI 生成的新内容
    # 使用正则匹配最后一个 "AI:" 后面的内容
    try:
        # 简单分割逻辑，实际可能需要更复杂的正则
        response = decoded.split("AI:")[-1].strip()
    except:
        response = decoded

    return response, history


# --- 6. Streamlit 界面逻辑 ---

st.title("🤖 智能旅游问答与目的地推荐系统")

# 6.1 侧边栏：用户偏好采集
with st.sidebar:
    st.header("🗺️ 旅游偏好采集")
    m_score = st.slider("山川地貌偏好", 1, 5, 3)
    o_score = st.slider("海洋地貌偏好", 1, 5, 3)
    p_score = st.slider("平原地貌偏好", 1, 5, 3)

    init_question = st.text_area("您对旅游还有其他问题吗？(可选)")

    start_btn = st.button("开始推荐")

# 初始化 Session State (用于存储对话历史和当前状态)
if "history" not in st.session_state:
    st.session_state["history"] = []
if "destination" not in st.session_state:
    st.session_state["destination"] = None

# 6.2 处理推荐点击
if start_btn:
    st.session_state["history"] = []  # 重置历史
    with st.spinner("AI正在分析最佳目的地..."):
        dest_cn, advice, img_b64, mime = recommend_destination(m_score, o_score, p_score)

        st.session_state["destination"] = dest_cn
        st.session_state["advice"] = advice
        st.session_state["img_data"] = (img_b64, mime)

        # 如果用户有初始问题，立即调用 LLM
        if init_question:
            # 构建一个包含 Context 的 Prompt
            context_prompt = (
                f"我根据您推荐的 {dest_cn} 很感兴趣。"
                f"我的地貌偏好是：山川{m_score}分，海洋{o_score}分，平原{p_score}分。"
                f"请基于推荐的目的地 {dest_cn} 回答我的问题：{init_question}"
            )
            resp, _ = predict(context_prompt, [])
            st.session_state["history"].append((init_question, resp))

# 6.3 展示推荐结果
if st.session_state["destination"]:
    st.divider()
    st.subheader(f"🎉 您的推荐目的地：{st.session_state['destination']}")
    st.info(st.session_state["advice"])

    img_b64, mime = st.session_state.get("img_data", (None, None))
    if img_b64:
        st.image(
            f"data:{mime};base64,{img_b64}",
            caption=f"{st.session_state['destination']} 景观示例",
            use_column_width=True
        )

# 6.4 历史对话展示
st.divider()
st.subheader("💬 对话历史")
for q, a in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# 6.5 新的对话输入
if st.session_state["destination"]:
    new_input = st.chat_input(f"关于 {st.session_state['destination']} 还有什么想问的？")
    if new_input:
        # 显示用户输入
        with st.chat_message("user"):
            st.write(new_input)

        with st.spinner("AI正在思考..."):
            # 调用模型，传入历史记录
            resp, _ = predict(new_input, st.session_state["history"])

            # 显示 AI 回答
            with st.chat_message("assistant"):
                st.write(resp)

            # 更新历史
            st.session_state["history"].append((new_input, resp))
            # 强制刷新页面以更新上面的历史列表 (Streamlit 特性)
            st.rerun()
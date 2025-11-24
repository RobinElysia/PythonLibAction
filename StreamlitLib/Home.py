import streamlit as st

st.write("Hello, world!")
st.markdown("## This is a header")

st.video("https://re.bluepoch.com/home/kv/p.mp4")

name = st.text_input('name', '请输入你的名字')

st.markdown("输入的内容：" + name)

age = st.number_input('age', 0, 100, 18)
st.markdown("输入的数字：" + str(age))

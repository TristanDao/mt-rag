import streamlit as st

st.title("Demo Streamlit")
st.write("Xin chào, đây là ứng dụng Streamlit đầu tiên!")

name = st.text_input("Nhập tên của bạn")
if name:
    st.success(f"Chào bạn {name} 👋")

age = st.slider("Chọn tuổi", 0, 100, 20)
st.write("Tuổi của bạn là:", age)

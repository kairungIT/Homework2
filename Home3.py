import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.header("kairung")
st.subheader("hengpraprohm")

html_8 = """
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>เกณฑ์การให้คะแนน 4</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

dx = pd.read_excel('./data/gen.xlsx')
st.dataframe(dx)

dt=pd.read_csv('./data/iris.csv')
st.dataframe(dt)

dt1 = dt['petal.length'].sum()
dt2 = dt['petal.width'].sum()
dt3 = dt['sepal.length'].sum()
dt4 = dt['sepal.width'].sum()
dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

st.balloons()

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   st.write(dt.head(20))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")


st.sidebar.markdown("# วิเคราะห์รายบุคคล ")

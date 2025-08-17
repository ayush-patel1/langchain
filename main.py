import streamlit as st
import langchain_helper
st.title("Resturant Finder")
cuisine=st.sidebar.selectbox("Pick a cuisine", ["Indian", "Chinese", "Italian","Mexican"])

if(cuisine):
   response=langchain_helper.generate_resturant_name_and_items(cuisine)
   st.header(response['resturant_name'])
   menu_items=response['menu_items']
   st.write(">>Menu Items:<<")
   for item in menu_items:
       st.write("-",item)
       st.write("-")

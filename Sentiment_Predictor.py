# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:03:43 2020

@author: HP
"""
import numpy as np
import pandas as pd
import pickle

import streamlit as st

import Functions_and_models


def predict_sentiments(lst):
    
    custom_inp_arr=np.array(lst)
    custom_inp_X=pd.DataFrame(custom_inp_arr,columns=['text'])

    prediction=saved_model.predict(custom_inp_X['text'])
   
    print(prediction)
    return prediction

saved_model=pickle.load(open('sentiment_analysis_model.pkl','rb'))
suggest_df=pd.read_excel('suggestions.xlsx')

def main():
    
    st.title("Sentiment Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Sentiment Predictor ML App </h2>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    
    num=st.slider("Number of input comments",min_value=1,max_value=10,value=5,step=1)
    

    
    no_sug=30
    lst=[]
    placeholder_lst=[]
    for i in range(num):
        if st.checkbox('Custom Input',value=False,key=i):
            inp=st.text_input('Input comment no {}'.format(i+1))
            place=st.empty()
        else:
            inp = st.selectbox('Input comment no {}'.format(i+1),suggest_df.iloc[i*no_sug:i*no_sug+no_sug,:]['Text'].tolist())
            place=st.empty()
        placeholder_lst.append(place)
        lst.append(inp)
 
    
    result=""
    if st.button("Predict"):
        result=predict_sentiments(lst)
        
        for i in range(num):
            if result[i]==0:
                placeholder_lst[i].markdown(":disappointed:**Negative**:-1:")
            elif result[i]==1:
                placeholder_lst[i].markdown(":smile:**Positive**:+1:") 
            else:
                placeholder_lst[i].markdown('Sentiment not predicted yet') 
            
    
        
    

    

if __name__=='__main__':
    main()
    

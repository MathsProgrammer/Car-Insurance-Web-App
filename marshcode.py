import streamlit as st
import pandas as pd
import pickle






st.title("Marshmellow Interview")

with st.sidebar:
    st.title("Sections")
    user_menu1 = st.radio("Navigation",
        options = ['Insurance Claim likelihood']
    )
#miles,age,exp,edu,inc,mar,credit,child,speeding,dui,past accidents
if user_menu1 == 'Insurance Claim likelihood':
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    q = st.slider("How many miles do you drive a year?", 100,1,100000)
    age = st.select_slider("How old are you?",["Young","Middle aged","Older"])
    if age == "Young":
        c = 1
    if age == "Middle aged":
        a = 1
    if age == "Older":
        b = 1        
    exp = st.select_slider("How experienced are you as a driver?",["Rookie","Amateur","Expert"])
    if exp == "Rookie":
        g = 1
    if exp == "Amateur":
        e = 1
    if exp == "Expert":
        f = 1 
    edu = st.selectbox("Did you go to university?", ["yes","no"])
    if edu == "yes":
        i = 1
    if edu == "no":
        h = 1
    inc = st.selectbox("How would you describe your income?", ["Low","Medium","High"])
    if inc == "Low":
        j = 1
    if inc == "Medium":
        l = 1
    if inc == "High":
        k = 1
    mar = st.selectbox("Are you married?", ["yes","no"])
    if mar == "yes":
        i = 0
    if mar == "no":
        i = 0
    m = st.number_input("What is your credit score?")
    chil = st.selectbox("Do you have children?", ["yes","no"])
    if chil == "yes":
        p = 0
    if chil == "no":
        p = 0
    s = st.slider("How DUI's have you have?", 1,1,3)
    r = st.slider("How many speeding tickets have you have?", 1,1,10)
    t = st.slider("How many accidents have you been in?", 1,1,10)
    
    d = {'Old':[a], 'Very Old':[b], 'Young':[c], 'male':[d], 'Amateur':[e], 'Expert':[f], 'Newbie':[g],
         'none':[h], 'university':[i], 'poverty':[j], 'upper class':[k], 'working class':[l],
         'CREDIT_SCORE':[m], 'VEHICLE_OWNERSHIP':[n], 'MARRIED':[o], 'CHILDREN':[p], 'ANNUAL_MILEAGE':[q],
         'SPEEDING_VIOLATIONS':[r], 'DUIS':[s], 'PAST_ACCIDENTS':[t]}
    
    
    df = pd.DataFrame(data=d)
    with open('model_pkl' , 'rb') as f:
        lr = pickle.load(f)
    k = lr.predict_proba(df)
    st.subheader("The probability that a claim will be made is:")
    st.success(k[0][1])


        








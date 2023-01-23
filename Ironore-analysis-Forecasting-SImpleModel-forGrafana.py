#!/usr/bin/env python
# coding: utf-8

# In[630]:


import pandas as pd
from prophet import Prophet
import streamlit as st


# In[631]:
st.header("Iron Ore Price Forecasting Tool")
st.write("Automated Forecast leveraging the historical price trend and seasonality")
st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
st.write("💡 The automated forecasting tool uses historical ASP published by IBM and Machine Learning to forecast the short term ASP" )
uploaded_file = st.file_uploader("Import Historical Data")
if uploaded_file is not None:
       dataframe = pd.read_csv(uploaded_file)

       #idf = pd.read_csv('C:\Python\ASP Prediction\PPS-IronOreW2.csv')
       idf=dataframe.copy()
       idf['Date'] = pd.to_datetime(idf['Date'], format='%Y-%m-%d')


       # In[632]:


       idfo=idf


       # In[633]:


       df=idf


       # In[634]:


       df['month'] = pd.to_datetime(df['Date'], format='%Y-%m')


       # In[635]:


       df.dropna(inplace=True)


       # In[636]:


       #idf.columns


       # In[637]:
       option = st.selectbox(
           'What Comodity you want to Predict',
           ('Lumps_Below 55%', 'Lumps_58 to 60%', 'Lumps_60 to 62%',
              'Lumps_62 to 65%', 'Lumps_Above 65%', 'Fines_Below 55%',
              'Fines_58 to 60%', 'Fines_60 to 62%', 'Fines_62 to 65%',
              'Fines_Above 65%', 'Iron Ore Conc.', 'Lumps_55 to 58%',
              'Fines_55 to 58%'))

       predictC=option
       #predictC=input("What Comodity you want to Predict")


       # In[638]:


       df = df.pivot(index='month', columns=predictC)

       start_date = df.index.min() - pd.DateOffset(day=1)
       end_date = df.index.max() + pd.DateOffset(day=31)
       dates = pd.date_range(start_date, end_date, freq='D')
       dates.name = 'Dates'
       df = df.reindex(dates, method='ffill')

       df = df.stack(predictC)
       #df = df.sortlevel(level=1)
       df = df.reset_index()


       # In[639]:


       dfo=df


       # In[640]:


       #len(df)


       # In[641]:


       idf=df


       # In[642]:


       idf1=idf.drop(columns=['Date'])


       # In[643]:


       idf1=idf.drop(columns=[predictC,'Dates','floor','cap'])


       # In[644]:


       inormalized_df=(idf1-idf1.min())/(idf1.max()-idf1.min())
       #inormalized_df=(idf1-idf1.mean())/idf1.std()


       # In[645]:


       #df=inormalized_df.join(idf[[predictC,'Dates','floor','cap']])
       df=inormalized_df.join(idf[[predictC,'Dates']])


       # In[646]:


       #len(df)


       # In[647]:


       #Fines_55 to 58%
       df1=df.rename(columns={"Dates": "ds",predictC: "y"})


       # In[648]:


       df1=df1.drop(columns=['Date'])


       # In[649]:


       #len(df1)


       # In[650]:


       m=Prophet(yearly_seasonality=True,
           weekly_seasonality=False,
           daily_seasonality=False,
                growth='linear',
                seasonality_mode='additive')


       # In[651]:


       df_predict3 = df1.drop(columns=['y']).copy()
       #df_train3=df1[:(1096-90)].copy()
       df_train3=df1.copy()

       # In[652]:


       m.fit(df_train3)
       future = m.make_future_dataframe(periods=120)
       #future
       #forecast= m.predict(df_predict3)
       forecast= m.predict(future)

       # In[653]:


       #dff=forecast[['yhat']].join(dfo[[predictC]])


       # In[654]:


       #forecast


       # In[655]:


       forecast['Date'] = pd.to_datetime(forecast['ds'], format='%Y-%m-%d')


       # In[656]:


       #forecast


       # In[657]:


       forecast1=forecast.set_index('Date')
       #forecast1


       # In[658]:


       dfr=forecast1.resample('MS').mean()


       # In[659]:


       dfg=dfr.reset_index().join(idfo[[predictC]])
       #st.write(dfg)


       # In[660]:
       st.write("Forecast for next 4 months")
       dfg['Date'] = pd.to_datetime(dfg['Date'], format='%Y-%m')
       #st.write(dfg)
       dfg1=dfg[dfg.isna().any(axis=1)]
       st.write(dfg1[['Date','yhat_lower','yhat','yhat_upper']])

       # In[665]:
       dfg2=pd.melt(dfg, id_vars =['Date'], value_vars =['yhat','yhat_upper','yhat_lower',predictC])
       #st.write(dfg2)
       #fname


       # In[666]:


       #from prophet.plot import add_changepoints_to_plot
       #fig = m.plot(forecast)
       #a = add_changepoints_to_plot(fig.gca(), m, forecast)
       #st.write(a)

       # In[ ]:



       st.write("Model Performance")
       import plotly.express as px
       fig=px.line(dfg2, x='Date', y='value', color='variable' )
       #fig = px.line(dfcsv, x=dfcsv['ReportedDate'], y=dfcsv['yhat'])
       st.plotly_chart(fig, use_container_width=True)

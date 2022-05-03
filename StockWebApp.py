# Description :   Stock Market Prediction Using ML
"""
Created on Mon April 21 2022
@author: MiteshRege
"""
# Importing Libraries
# from email import header
# from pydoc import describe
# from nbformat import write
# from pandas_datareader import get_data_enigma
# from pyrsistent import T
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from matplotlib import pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# import datetime import date
# from prophet import Prophet
# from fbprophet import Prophet
# import fbprophet
# from fbprophet.plot import plot_plotly 
from plotly import graph_objs as go

# for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


# from torch import set_deterministic_debug_mode
# import pan
# data.DataReader("AAPL",start='2015-1-1',end='2015-12-31',data_source='yahoo')

# Add a title and an image




def main():

    st.write("""
    # Stock Market Prediction Using Machine Learning \n
    ### Welcome to our stock prediction web app 
      Xavier Institute Of Engineering   TE IT  
    """)
    video_file = open("C:/Users/Mitesh Rege/Desktop/Project_stock/video/video.mp4",'rb')
    video_bytes = video_file.read()
    st.video(video_file)

    image1 = Image.open("C:/Users//Mitesh Rege/Desktop/Project_stock/stockimage1.jpg")
    st.image(image1,use_column_width=True)


    # Creating a sidebar header
    st.sidebar.header("User Input")
    st.sidebar.subheader("Enter all below input fields ")
    # Create a func to get user imput
    def get_input():
        start_date   = st.sidebar.text_input("Start Date","2020-01-02 00:00:00+00:00")
        end_date     = st.sidebar.text_input("End Date","2022-04-21 00:00:00+00:00")
        # stock_symbol = st.sidebar.text_input("Stock Symbol","")
        stock_symbol  = st.sidebar.selectbox("Select Company",("AMZN","GOOG","INFY","TSLA"))
        st.sidebar.write("Stock Investment Are Subjected to market risks ,read all scheme related documents carefully")
        return start_date,end_date,stock_symbol

        
    # Creating a function to get the company name 
    def get_company_name(symbol):
        if symbol == "AMZN":
            return 'Amazon'
        elif symbol == 'TSLA':
            return 'TESLA'
        elif symbol ==  'GOOG':
            return 'Google'
        elif symbol ==  'INFY':
            return 'Infosys'
        else:
            'None'

    # create a function to get the proper company data and proper timeframe from the user start date , end date imputs
    def get_data(symbol,start,end):

        #load the data 
        if symbol.upper() == 'AMZN':
            df = pd.read_csv("C:/Users/Mitesh Rege/Desktop/Project_stock/Datasets/AMZN.csv")
            df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
            df.index = df['date']
        elif symbol.upper() == 'TSLA':
            df = pd.read_csv("C:/Users/Mitesh Rege/Desktop/Project_stock/Datasets/TSLA.csv")
            df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
            df.index = df['date']
        elif symbol.upper() == 'GOOG':
            df = pd.read_csv("C:/Users/Mitesh Rege/Desktop/Project_stock/Datasets/GOOG.csv")
            df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
            df.index = df['date']
        elif symbol.upper() == 'INFY':
            df = pd.read_csv("C:/Users/Mitesh Rege/Desktop/Project_stock/Datasets/INFY.csv")
            df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
            df.index = df['date']
        else:
            df = pd.DataFrame(columns= ['symbol','date','close','high','low','open','volume','adjClose','adjHigh','adjLow','adjOpen','adjVolume','divCash','splitFactor'])
            df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
            df.index = df['date']

        # Get the date range
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        #set the start and end index rows both to 
        start_row =  0 
        end_row   =  0

        #Start the date from the top ot the data set and go down to see if user start date is less than or equalto the date in dataset
        # stock market may be closed that day becz of holiday and user enterd that dates 

        for i in range(0,len(df)):
            if start <= pd.to_datetime(df['date'][i]):
                start_row =  i 
                break
        #start from the botto of date set 
        # check user end date enterd by user is not greter than we have it in dateframe
        for j in range(0,len(df)):
            if end>=pd.to_datetime(df['date'][len(df)-1-j]):
                end_row = len(df)- 1- j
                break
        
        #set the index to be the date 
        df= df.set_index(df['date'].values) 
        # df= df.set_index(pd.DatetimeIndex(df['date'].values))

        return df.iloc[start_row:end_row +1 , :]

    #Get the users input
    start , end , symbol = get_input()

    #Get the data 
    df = get_data(symbol,start,end) 
    # Get the company name
    company_name = get_company_name(symbol.upper())
    
    


    
    # Display the close value
    st.header(company_name+" Open Price\n")
    st.line_chart(df[['open']])

    # Display the volume 
    st.header(company_name+" Close Price\n")
    st.line_chart(df['close'])
    #get statistics on the data
    st.header(company_name+' Data Statistics ')
    st.write(df.describe())

    # Raw Data
    st.subheader(company_name+' Raw Data')
    st.write(df.tail())
    
    st.subheader(company_name+' Coorelation Graph Of All Attributes Present in the dataset')
    st.write(df.corr())
    # Prophet
    # prophet start//////
    # ploting the raw data
                            # def plot_raw_data():
                            #     fig = go.Figure() 
                            #     fig.add_trace(go.Scatter(x=df['date'],y=df['open'],name="Stock_open"))
                            #     fig.add_trace(go.Scatter(x=df['date'],y=df['close'],name="Stock_close"))
                            #     fig.layout.update(title_text=' Time Series Date ',xaxis_rangeslider_visible=True)
                            #     st.plotty_chart(fig)
                            # plot_raw_data()

                            # # future forcast using
                            # df_train = df[['date','close']]
                            
                            # m = Prophet()
                            # m.fit(df_train)
                            # n_year = st.slider("Enter Years of Predicion ",1,4)
                            # period = n_year*365
                            # future = m.make_future_dataframe(periods=period)
                            # forecast = m.predict(future)

                            # st.subheader('Forecast Data')
                            # st.write(forecast.tail())

                            # fig1 = plot_plotly(m,forecast)
                            # st.plottly_chart(fig1)

                            # st.write('forcast components')
                            # fig2 = m.plot_components(forecast)
                            # st.write(fig2)


    # prophet end//////

    # Code For Linear Regression  & Decision Tree Prediction  ::--

    st.write(" ### This is LR Predicting Next 'N' daysStockPrice where user will enter 'N ' ")
    st.write(" Below Enter the N values , N are No.of Future Days")
    future_days = st.text_input(" Below  Enter For How Many Future Days You Want To Predict Stock Price","30")
    future_days = int(future_days)

    df_infy_open = pd.DataFrame(df['open'])
    df_infy_open['Prediction'] = df_infy_open['open'].shift(-future_days)
    # Creating a feature data set (X) and convert it to a numpy array  and remove the last 'x' rows/day's
    X = np.array(df_infy_open.drop(['Prediction'], 1))[:-future_days]
    # Creating a target data set (y) and convert it to a numpy array  and to get the target values except the last 'x' rows/day's
    y = np.array(df_infy_open['Prediction'])[:-future_days]
    
    # Predicting future 30 days with an Linear and Decision Tree
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # Implementing Linear and Decision Tree Regression Algorithms.
    tree = DecisionTreeRegressor().fit(x_train, y_train)
    lr = LinearRegression().fit(x_train, y_train)
    # Get the last 'x' rows of the feature data set 
    x_future = df_infy_open.drop(['Prediction'], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    tree_prediction = tree.predict(x_future)
    # print(tree_prediction)  # it is predicted stock price of nxt 30 days by DT

    predictions = tree_prediction 
    valid = df_infy_open[X.shape[0]:]
    valid['Predictions_dt'] = predictions

    lr_prediction = lr.predict(x_future)
    # print(lr_prediction)  # it is predicted stock price of nxt 30 days by LR


    predictions_lr = lr_prediction 
    valid = df_infy_open[X.shape[0]:]
    valid['Predictions_lr'] = predictions_lr

    # plt.figure(figsize=(16,8))
    # plt.title("Model")
    # plt.xlabel('Days')
    # plt.ylabel('Open Price ')
    # plt.plot(df_infy_open['open'])
    # plt.plot(valid[['open', 'Predictions_dt']])
    # plt.legend(["Original", "Valid", 'Predicted Using DT'])
    # plt.show()
    st.header(company_name+" Next "+str(future_days)+" days Prediction Using Linear Regression \n")
    valid['Predictions_lr'] = predictions_lr
    st.line_chart(valid.Predictions_lr.values)



    st.header(company_name+" Next "+str(future_days)+" days Prediction Using Decision Tree \n")
    valid['Predictions_dt'] = predictions
    st.line_chart(valid.Predictions_dt.values)
     # st.pyplot(valid[['open', 'Predictions_dt']])
   
    st.write('# Time Series Analysis ')

    st.write("## Simple Moving Average ")
    st.write("It is majorally used in stock market short-term trading in-which when our SMA( i.e smooth curve ) cuts the original line/graph then the trader can make decision of buying/sellings the stocks")
    
    # Simple moving average
    # ## .rolling(window-size,minimum-period)
    # - window-size    :-  5  - Icluding that value and above  it will take mean of 5 then replace there
    # - minimum-period :-  5  - starting  4 values become nan

    # Modifications to the data or indices of the copy will not be reflected in the original object

    df_infy = df.copy() 

    df_infy['open:10 days rooling '] = df_infy['open'].rolling(window=10,min_periods=2).mean()
    df_infy['open:30 days rooling '] = df_infy['open'].rolling(window=30,min_periods=2).mean()
    
    # Lets plot for above cols
    # Jan Month of 2021
    # con = df_infy[['open','open:10 days rooling ','open:30 days rooling ']].plot(xlim=['2022-01-01','2022-04-19'])
    # Here its not clearly visible  so , look into next cell output
    st.line_chart(df_infy[['open','open:10 days rooling ','open:30 days rooling ']])
    
    st.write("## Eponential Moving Average ")
    st.write("EMA is a type of moving average (MA) that places a greater weight and significance on the most recent data points.")
    st.write("""
    formula : ((close_value - prev_ema_value )*multiplier ) + prev_ema_value
    formula for multiplier: (2/( Window_rolling_size_value )+1)
    Since its req prev ema value so
    for 1st value we need to do sma with cosidering rolling_window_size same as to used while calculating multiplier


    """)
   
    df_infy['EMA_0.1'] = df_infy['open'].ewm(alpha=0.1,adjust=False).mean()
    # lets do for diff  smothening factor
    df_infy['EMA_0.2'] = df_infy['open'].ewm(alpha=0.2,adjust=False).mean()
    df_infy['EMA_0.3'] = df_infy['open'].ewm(alpha=0.3,adjust=False).mean()
    st.line_chart( df_infy[['EMA_0.1','EMA_0.2','EMA_0.3']])
    # df_infy[['EMA_0.1','EMA_0.2','EMA_0.3']]

   
    st.write("## Eponential Weighted  Moving Average ")
    st.write("EWMA is s a quantitative or statistical measure used to model or describe a time series. The EWMA is widely used in finance, the main applications being technical analysis and volatility modeling")
    st.write("""
    Formula :
    EMWA(t) = a* x(t) + (1-a)*EMWA(t-1)

    here a => weight (providing more imp for current data) & to prevent lag

    """)

    # Lets plot EWMA
    df_infy['EWMA'] = df_infy['open'].ewm(span=5).mean()
    st.write("Plotting the EWMA Curve ")
    st.line_chart(df_infy[['open','EWMA']])

    # NOW KNN Starts 

    df_infy_open = pd.DataFrame(df['open'])
    df_infy_open['Prediction'] = df_infy_open['open'].shift(-future_days)
    # Creating a feature data set (X) and convert it to a numpy array  and remove the last 'x' rows/day's
    X = np.array(df_infy_open.drop(['Prediction'], 1))[:-future_days]
    # Creating a target data set (y) and convert it to a numpy array  and to get the target values except the last 'x' rows/day's
    y = np.array(df_infy_open['Prediction'])[:-future_days]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    #scaling data
    # x_valid = valid.drop('open', axis=1)
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)
    # x_valid_scaled = scaler.fit_transform(x_test)
    # x_valid = pd.DataFrame(x_valid_scaled)

    #using gridsearch to find the best parameter
    params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)

    #fit the model and make predictions
    model.fit(x_train,y_train)

    # Get the last 'x' rows of the feature data set 
    x_future = df_infy_open.drop(['Prediction'], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    # predicting     
    predictions_knn = model.predict(x_future)
    valid = df_infy_open[X.shape[0]:]
    valid['Predictions_knn'] = predictions_knn
    
    st.header(company_name+" Next "+str(future_days)+" days Prediction Using KNN Regressor \n")
    st.line_chart(valid.Predictions_knn)
    
    st.write(" Image Showing Future Prediction By An Output Produced By KNN Algorithm in Python")
    image2 = Image.open("C:/Users/Mitesh Rege/Desktop/Project_stock/stocl_image_knn.png")
    st.image(image2,use_column_width=True)
    
    st.write(" Image Showing Future Prediction By An Output Produced By Linear Regression Algorithm in Python")
    image_lr = Image.open("C:/Users/Mitesh Rege/Desktop/Project_stock/stocl_image_lr.png")
    st.image(image_lr,use_column_width=True)
    
    st.write(" Image Showing Future Prediction By An Output Produced By Decision Tree Algorithm in Python")
    image_dt= Image.open("C:/Users/Mitesh Rege/Desktop/Project_stock/stocl_image_dt.png")
    st.image(image_dt,use_column_width=True)


    
if __name__=='__main__': 
    main()
    st.write(" ## Thank You For Visiting ! ")
    st.write(" ## Please Visit Again ")
    st.write("""
           ### Team Members   : 
             
             1. Mitesh Rege   
             2. Shivam Mishra   
             3. Prajna Shetty   
              """)
    st.write(" ### Any Queries :  ")
    st.write(" Contact  : \n Mitesh Rege  \n Email Address : 201903038.miteshrpa@student.xavier.ac.in  ")
    st.write("  Xavier Institute Of Engineering  Mahim Mumbai ")
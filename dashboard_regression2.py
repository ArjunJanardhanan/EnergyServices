
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import pickle
from sklearn import  metrics
import numpy as np

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Load data
df = pd.read_csv('IST_Central_2019_test.csv')
#df['Date'] = pd.to_datetime (df['Date']) # create a new column 'data time' of datetime type

df2=df.iloc[:,2:7]
X2=df2.values
# fig1 = px.line(df, x="date", y=df.columns[1:4])# Creates a figure with the raw data

feature = df.columns[1:]


#df_real = pd.read_csv('real_results.csv')
y2=df['Power[kW]'].values

#Load and run FR model

with open('Lregr.pkl','rb') as file:
    LR_model=pickle.load(file)

y2_pred_LR = LR_model.predict(X2)

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR) 
MBE_LR=np.mean(y2-y2_pred_LR)
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)
NMBE_LR=MBE_LR/np.mean(y2)
#Load RF model
with open('RF_model.pkl','rb') as file:
    RF_model=pickle.load(file)

y2_pred_RF = RF_model.predict(X2)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF)
MBE_RF=np.mean(y2-y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)
NMBE_RF=MBE_RF/np.mean(y2)

# Create data frames with predictin results and error metrics 
d = {'Methods': ['Linear Regression','Random Forest'], 'MAE': [MAE_LR, MAE_RF],'MBE': [MBE_LR, MBE_RF], 'MSE': [MSE_LR, MSE_RF], 'RMSE': [RMSE_LR, RMSE_RF],'cvMSE': [cvRMSE_LR, cvRMSE_RF],'NMBE': [NMBE_LR, NMBE_RF]}
df_metrics = pd.DataFrame(data=d)
d={'date':df['date'], 'LinearRegression': y2_pred_LR,'RandomForest': y2_pred_RF}
df_forecast=pd.DataFrame(data=d)

regression = df_forecast.columns[1:]

# merge real and forecast results and creates a figure with it
df=pd.merge(df,df_forecast, on='date')
# fig2 = px.line(df_forecast,x=df_forecast.columns[0],y=df_forecast.columns[1:])

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ],style={'margin': 'auto'})


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server=app.server
app.layout = html.Div([
    html.H1('Energy Forecast Tool', style={'text-align': 'center'}),
    html.P('Visualizing real data and forecasting electricity consumption in the central building of IST from January to March 2019 with error metrics', style={'text-align': 'center'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Error Metrics', value='tab-3'),
    ]),
    html.Div(id='tabs-content'),
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('Raw data for the central building of IST (January to March 2019)', style={'text-align': 'center'}),
            html.Label('Select the variables to add in the plot below:'),
            dcc.Checklist(
                id='feature_checklist',
                options=[{'label': i, 'value': i} for i in feature],
                value=['Power[kW]'],
                inline=True
            ),
            dcc.Graph(
                id='yearly-data'
            ),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H4('Predicted and real data for the central building of IST (January to March 2019)', style={'text-align': 'center'}),
            html.Label('Select the regression method to add in the plot below:'),
            dcc.Checklist(
                id='forecast_checklist',
                options=[{'label': i, 'value': i} for i in regression],
                value=['RandomForest'],
                inline=True
            ),
            html.Br(),
            html.Label('Select the real data/features to add in the plot below:'),
            dcc.Checklist(
                id='real_checklist',
                options=[{'label': i, 'value': i} for i in feature],
                value=['Power[kW]'],
                inline=True
            ),
            dcc.Graph(
                id='forecast_data'
                # figure=fig2,
                ), 
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H4('Error metrics for different regression methods', style={'text-align': 'center'}),
                        generate_table(df_metrics)
        ])

@app.callback(
    Output('yearly-data', 'figure'),
    [Input('feature_checklist', 'value')]
)
def update_graph1(selected_features):
    traces = []
    for feature in selected_features:
        # trace = px.line(df, x="date", y=selected_features)
        trace = go.Scatter(x=df["date"], y=df[feature], mode='lines', name=feature)
        traces.append(trace)
    # Create the figure object
    fig = {'data': traces, 'layout': {'title': 'Selected Features'}}
    return fig

@app.callback(
    dash.dependencies.Output('forecast_data', 'figure'),
    [dash.dependencies.Input('forecast_checklist', 'value'),
     dash.dependencies.Input('real_checklist', 'value'),
      ]
)
def update_graph2(selected_features,selected_feature):
    traces = []
    for feature in selected_features:
        trace = go.Scatter(x=df["date"], y=df[feature], mode='lines', name=feature)
        traces.append(trace)      
    for feature in selected_feature:
        trace = go.Scatter(x=df["date"], y=df[feature], mode='lines', name=feature)
        traces.append(trace)   
    # Create the figure object
    fig = {'data': traces, 'layout': {'title': 'Selected Features'}}
    return fig


if __name__ == '__main__':
    app.run_server()


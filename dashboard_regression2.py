
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

from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Load data
df = pd.read_csv('IST_Central_2019_test.csv')
#df['Date'] = pd.to_datetime (df['Date']) # create a new column 'data time' of datetime type

df2=df.iloc[:,2:7]
X2=df2.values
# fig1 = px.line(df, x="date", y=df.columns[1:4])# Creates a figure with the raw data
df1=df.iloc[:,2:]
X1=df1.values

feature = df.columns[1:]
feat=df1.columns

#df_real = pd.read_csv('real_results.csv')
y2=df['Power[kW]'].values

# Define feature selection methods
def filter_method(X1, y2):
    features = SelectKBest(k=5, score_func=mutual_info_regression)
    fit = features.fit(X1, y2)
    return fit.scores_

def wrapper_method(X1, y2):
    model = LinearRegression() 
    rfe = RFE(model,n_features_to_select=3)
    fit = rfe.fit(X1, y2)
    return fit.ranking_

def ensemble_method(X1, y2):
    model = RandomForestRegressor()
    model.fit(X1, y2)
    return model.feature_importances_





#Load and run LR model
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

#Load and run SV model
with open('Sregr.pkl','rb') as file:
    SV_model=pickle.load(file)

y2_pred_SV = SV_model.predict(X2)

#Evaluate errors
MAE_SV=metrics.mean_absolute_error(y2,y2_pred_SV) 
MBE_SV=np.mean(y2-y2_pred_SV)
MSE_SV=metrics.mean_squared_error(y2,y2_pred_SV)  
RMSE_SV= np.sqrt(metrics.mean_squared_error(y2,y2_pred_SV))
cvRMSE_SV=RMSE_SV/np.mean(y2)
NMBE_SV=MBE_SV/np.mean(y2)

#Load and run DT model
with open('DT_regr_model.pkl','rb') as file:
    DT_model=pickle.load(file)

y2_pred_DT = DT_model.predict(X2)

#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y2,y2_pred_DT) 
MBE_DT=np.mean(y2-y2_pred_DT)
MSE_DT=metrics.mean_squared_error(y2,y2_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y2,y2_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y2)
NMBE_DT=MBE_DT/np.mean(y2)

#Load and run GB model
with open('GB_model.pkl','rb') as file:
    GB_model=pickle.load(file)

y2_pred_GB = GB_model.predict(X2)

#Evaluate errors
MAE_GB=metrics.mean_absolute_error(y2,y2_pred_GB) 
MBE_GB=np.mean(y2-y2_pred_GB)
MSE_GB=metrics.mean_squared_error(y2,y2_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y2,y2_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y2)
NMBE_GB=MBE_GB/np.mean(y2)

#Load and run XGB model
with open('XGB_model.pkl','rb') as file:
    XGB_model=pickle.load(file)

y2_pred_XGB = XGB_model.predict(X2)

#Evaluate errors
MAE_XGB=metrics.mean_absolute_error(y2,y2_pred_XGB) 
MBE_XGB=np.mean(y2-y2_pred_XGB)
MSE_XGB=metrics.mean_squared_error(y2,y2_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y2,y2_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y2)
NMBE_XGB=MBE_XGB/np.mean(y2)

#Load and run BT model
with open('BT_model.pkl','rb') as file:
    BT_model=pickle.load(file)

y2_pred_BT = BT_model.predict(X2)

#Evaluate errors
MAE_BT=metrics.mean_absolute_error(y2,y2_pred_BT) 
MBE_BT=np.mean(y2-y2_pred_BT)
MSE_BT=metrics.mean_squared_error(y2,y2_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y2,y2_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y2)
NMBE_BT=MBE_BT/np.mean(y2)


#Load and run NN model
with open('NN_model.pkl','rb') as file:
    NN_model=pickle.load(file)

y2_pred_NN = NN_model.predict(X2)

#Evaluate errors
MAE_NN=metrics.mean_absolute_error(y2,y2_pred_NN) 
MBE_NN=np.mean(y2-y2_pred_NN)
MSE_NN=metrics.mean_squared_error(y2,y2_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y2,y2_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y2)
NMBE_NN=MBE_NN/np.mean(y2)

# Create data frames with predictin results and error metrics 
d = {'Methods': ['Linear Regression','Random Forest Regression','Support Vector Regression','Decision Tree Regression','Gradient Boosting Regression','Extreme Gradient Boosting Regression','Bootstrapping Regression', 'Neural Network Regression'], 'MAE': [MAE_LR, MAE_RF, MAE_SV, MAE_DT, MAE_GB, MAE_XGB, MAE_BT, MAE_NN],'MBE': [MBE_LR, MBE_RF, MBE_SV, MBE_DT, MBE_GB, MBE_XGB, MBE_BT, MBE_NN], 'MSE': [MSE_LR, MSE_RF, MSE_SV, MSE_DT, MSE_GB, MSE_XGB, MSE_BT, MSE_NN], 'RMSE': [RMSE_LR, RMSE_RF, RMSE_SV, RMSE_DT, RMSE_GB, RMSE_XGB, RMSE_BT, RMSE_NN],'cvMSE': [cvRMSE_LR, cvRMSE_RF, cvRMSE_SV, cvRMSE_DT, cvRMSE_GB, cvRMSE_XGB, cvRMSE_BT, cvRMSE_NN],'NMBE': [NMBE_LR, NMBE_RF, NMBE_SV, NMBE_DT, NMBE_GB, NMBE_XGB, NMBE_BT, NMBE_NN]}
df_metrics = pd.DataFrame(data=d)
d={'date':df['date'], 'Linear Regression': y2_pred_LR,'Random Forest Regression': y2_pred_RF,'Support Vector Regression': y2_pred_SV,'Decision Tree Regression': y2_pred_DT,'Gradient Boosting Regression': y2_pred_GB,'Extreme Gradient Boosting Regression': y2_pred_XGB,'Bootstrapping Regression': y2_pred_BT, 'Neural Network Regression': y2_pred_NN}
df_forecast=pd.DataFrame(data=d)

regression = df_forecast.columns[1:]

# merge real and forecast results and creates a figure with it
df=pd.merge(df,df_forecast, on='date')
df=df.dropna()
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
        dcc.Tab(label='Exploratory Data Analysis', value='tab-4'),
        dcc.Tab(label='Feature Selection', value='tab-5')
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
                value=['Random Forest Regression'],
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
    elif tab == 'tab-4':
        return html.Div([
            html.H4('Exploratory analysis of real data', style={'text-align': 'center'}),
            html.Label('Select the variable:'),
            dcc.RadioItems(
                id='feature-radio',
                options=[{'label': f, 'value': f} for f in feature],
                value=feature[0],  # Default value
                inline=True
            ),
            html.Br(),
            html.Label('Select the plot type:'),
            dcc.Dropdown(id='plot-type',
                         options=[{'label': 'Histogram', 'value': 'histogram'},
                                  {'label': 'Boxplot', 'value': 'boxplot'}],
                         value='histogram'), 
            dcc.Graph(id='plot')
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.H4('Visualizing the significance of features in modeling',style={'text-align': 'center'}),
            html.Label('Select the feature selection method:'),           
            dcc.Dropdown(
                id='feature-method-dropdown',
                options=[
                    {'label': 'Filter Method (KBest)', 'value': 'filter'},
                    {'label': 'Wrapper Method (RFE)', 'value': 'wrapper'},
                    {'label': 'Ensemble Method', 'value': 'ensemble'}
                ],
                value='filter',
            ),
            html.Div(id='feature-selection-results')
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
    fig = {'data': traces, 'layout': {'title': 'Selected plots'}}
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
    fig = {'data': traces, 'layout': {'title': 'Selected plots'}}
    return fig

# Callback to update histogram and boxplot based on selected feature
@app.callback(
    Output('plot', 'figure'),
    [Input('feature-radio', 'value'),
     Input('plot-type', 'value')]
)
def update_plots(selected_feature, plot_type):
    
    if plot_type == 'histogram':
        # Create histogram
        fig3 = px.histogram(df, x=selected_feature, title=f'Histogram for {selected_feature}')
    else:
        # Create boxplot
        fig3 = px.box(df, y=selected_feature, title=f'Boxplot for {selected_feature}')
    return fig3

@app.callback(
    Output('feature-selection-results', 'children'),
    Input('feature-method-dropdown', 'value')
)
def perform_feature_selection(method):
    if method == 'filter':
        scores = filter_method(X1, y2)
        return html.Div([
            dcc.Graph(figure={
                'data': [go.Bar(x=feat, y=scores)],
                'layout': {'title': 'Feature Importance Scores for Filter Method'}
            })
        ])
    elif method == 'wrapper':
        rankings = wrapper_method(X1, y2)
        return html.Div([
            dcc.Graph(figure={
                'data': [go.Bar(x=feat, y=rankings)],
                'layout': {'title': 'Feature Rankings for Wrapper Method'}
            })
        ])
       
    elif method == 'ensemble':
        importances = ensemble_method(X1, y2)
        return html.Div([
            dcc.Graph(figure={
                'data': [go.Bar(x=feat, y=importances)],
                'layout': {'title': 'Feature Importances for Ensemble Method'}
            })
        ])

if __name__ == '__main__':
    app.run_server(debug=False)


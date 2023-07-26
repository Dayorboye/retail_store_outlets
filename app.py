
import dash
import datetime
from dash.dependencies import Input, Output, State
from dash import Input, Output, dcc, html,Dash,dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import dash_daq as daq
import plotly.graph_objs as go
import app
import numpy as np
import dash_table
from dash_table.Format import Format, Group
import dash_table.FormatTemplate as FormatTemplate
from datetime import datetime as dt
import datetime
import base64
import io

###################################
# Import libaries and create model function
##################################
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


####################################################################################################
# 000 - BANNER AND TAB
####################################################################################################
tabHead = dbc.Nav(
    [   dbc.NavLink("Sales Prediction", href="/", active="exact"),
        dbc.NavLink("Abuja Branch Charts", href="/Abuja_Branch_Control_Charts", active="exact"),
        dbc.NavLink("Lagos Branch Charts", href="/Lagos_Branch_Control_Charts", active="exact"),
        dbc.NavLink("PHcourt Branch Charts", href="/Portharcourt_Branch_Control_Charts", active="exact"),
        dbc.NavLink("Decision Engine", href="/Management_Decision_Engine", active="exact"),  
    ],style={'width': '80%','margin-bottom':'-3px' },
)


df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")

# Iris bar figure
def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H1("Store Performance Dashboard" ,style={'textAlign': 'center'}),
                ]
            ),
        ],
    )


url = "XYZ_Company_Dataset.csv"
XYZ_Company_Dataset = pd.read_csv(url)

XYZ_cogs_sum = 'Goods Cost'
XYZ_Company_cogs_sum = XYZ_Company_Dataset['cogs'].sum().round()
XYZ_gR_sum = 'Gross Revenue'
XYZ_Company_gR_sum = XYZ_Company_Dataset['Gross Revenue'].sum().round()
XYZ_gInc_sum = 'Gross Income'
XYZ_Company_gInc_sum = XYZ_Company_Dataset['gross income'].sum().round()
XYZ_gR_M_sum  = 'Member Revenue'
XYZ_Company_gR_M_sum  = XYZ_Company_Dataset.loc[XYZ_Company_Dataset['Customer type'] == 'Member', ['Gross Revenue']].sum().round()[0]
XYZ_gInc_M_sum  = 'Member Income'
XYZ_Company_gInc_M_sum  = XYZ_Company_Dataset.loc[XYZ_Company_Dataset['Customer type'] == 'Member', ['gross income']].sum().round()[0]
XYZ_gR_N_sum = 'Normal Revenue'
XYZ_Company_gR_N_sum = XYZ_Company_Dataset.loc[XYZ_Company_Dataset['Customer type'] == 'Normal', ['Gross Revenue']].sum().round()[0]
XYZ_gInc_N_sum = 'Normal Income'
XYZ_Company_gInc_N_sum = XYZ_Company_Dataset.loc[XYZ_Company_Dataset['Customer type'] == 'Normal', ['gross income']].sum().round()[0]
XYZ_M_T_sum = 'Member {:.0%} Tax'.format(.05)
XYZ_Company_M_T_sum = XYZ_Company_Dataset.loc[XYZ_Company_Dataset['Customer type'] == 'Member', ['Tax 5%']].sum().round()[0]
XYZ_N_T_sum = 'Normal {:.0%} Tax'.format(.05)
XYZ_Company_N_T_sum  = XYZ_Company_Dataset.loc[XYZ_Company_Dataset['Customer type'] == 'Normal', ['Tax 5%']].sum().round()[0]




####################################################################################################
# 000 - IMPORT DATASET FOR ABUJA BRANCH
####################################################################################################
url1 = "Abuja_branch_dataset.csv"
Abuja_branch_dataset = pd.read_csv(url1)

Ab_cogs_sum = 'Goods Cost'
Abuja_cogs_sum = Abuja_branch_dataset['cogs'].sum().round()
Ab_gR_sum = 'Gross Revenue'
Abuja_gR_sum = Abuja_branch_dataset['Gross Revenue'].sum().round()
Ab_gInc_sum = 'Gross Income'
Abuja_gInc_sum = Abuja_branch_dataset['gross income'].sum().round()
Ab_gR_M_sum  = 'Member Revenue'
Abuja_gR_M_sum  = Abuja_branch_dataset.loc[Abuja_branch_dataset['Customer type'] == 'Member', ['Gross Revenue']].sum().round()[0]
Ab_gInc_M_sum  = 'Member Income'
Abuja_gInc_M_sum  = Abuja_branch_dataset.loc[Abuja_branch_dataset['Customer type'] == 'Member', ['gross income']].sum().round()[0]
Ab_gR_N_sum = 'Normal Revenue'
Abuja_gR_N_sum = Abuja_branch_dataset.loc[Abuja_branch_dataset['Customer type'] == 'Normal', ['Gross Revenue']].sum().round()[0]
Ab_gInc_N_sum = 'Normal Income'
Abuja_gInc_N_sum = Abuja_branch_dataset.loc[Abuja_branch_dataset['Customer type'] == 'Normal', ['gross income']].sum().round()[0]
Ab_M_T_sum = 'Member {:.0%} Tax'.format(.05)
Abuja_M_T_sum = Abuja_branch_dataset.loc[Abuja_branch_dataset['Customer type'] == 'Member', ['Tax 5%']].sum().round()[0]
Ab_N_T_sum = 'Normal {:.0%} Tax'.format(.05)
Abuja_N_T_sum  = Abuja_branch_dataset.loc[Abuja_branch_dataset['Customer type'] == 'Normal', ['Tax 5%']].sum().round()[0]


####################################################################################################
# 000 - IMPORT DATASET FOR LAGOS BRANCH
####################################################################################################
url2 = "Lagos_branch_dataset.csv"
Lagos_branch_dataset = pd.read_csv(url2)


La_cogs_sum = 'Goods Cost'
Lagos_cogs_sum = Lagos_branch_dataset['cogs'].sum().round()
La_gR_sum = 'Gross Revenue'
Lagos_gR_sum = Lagos_branch_dataset['Gross Revenue'].sum().round()
La_gInc_sum = 'Gross Income'
Lagos_gInc_sum = Lagos_branch_dataset['gross income'].sum().round()
La_gR_M_sum  = 'Member Revenue'
Lagos_gR_M_sum  = Lagos_branch_dataset.loc[Lagos_branch_dataset['Customer type'] == 'Member', ['Gross Revenue']].sum().round()[0]
La_gInc_M_sum  = 'Member Income'
Lagos_gInc_M_sum  = Lagos_branch_dataset.loc[Lagos_branch_dataset['Customer type'] == 'Member', ['gross income']].sum().round()[0]
La_gR_N_sum = 'Normal Revenue'
Lagos_gR_N_sum = Lagos_branch_dataset.loc[Lagos_branch_dataset['Customer type'] == 'Normal', ['Gross Revenue']].sum().round()[0]
La_gInc_N_sum = 'Normal Income'
Lagos_gInc_N_sum = Lagos_branch_dataset.loc[Lagos_branch_dataset['Customer type'] == 'Normal', ['gross income']].sum().round()[0]
La_M_T_sum = 'Member {:.0%} Tax'.format(.05)
Lagos_M_T_sum = Lagos_branch_dataset.loc[Lagos_branch_dataset['Customer type'] == 'Member', ['Tax 5%']].sum().round()[0]
La_N_T_sum = 'Normal {:.0%} Tax'.format(.05)
Lagos_N_T_sum  = Lagos_branch_dataset.loc[Lagos_branch_dataset['Customer type'] == 'Normal', ['Tax 5%']].sum().round()[0]



####################################################################################################
# 000 - IMPORT DATASET FOR PORTHARCOURT BRANCH
####################################################################################################
url3 = "PortHarcourt_branch_dataset.csv"
PortHarcourt_branch_dataset = pd.read_csv(url3)

Ph_cogs_sum = 'Goods Cost'
PortHarcourt_cogs_sum = PortHarcourt_branch_dataset['cogs'].sum().round()
Ph_gR_sum = 'Gross Revenue'
PortHarcourt_gR_sum = PortHarcourt_branch_dataset['Gross Revenue'].sum().round()
Ph_gInc_sum = 'Gross Income'
PortHarcourt_gInc_sum = PortHarcourt_branch_dataset['gross income'].sum().round()
Ph_gR_M_sum  = 'Member Revenue'
PortHarcourt_gR_M_sum  = PortHarcourt_branch_dataset.loc[PortHarcourt_branch_dataset['Customer type'] == 'Member', ['Gross Revenue']].sum().round()[0]
Ph_gInc_M_sum  = 'Member Income'
PortHarcourt_gInc_M_sum  = PortHarcourt_branch_dataset.loc[PortHarcourt_branch_dataset['Customer type'] == 'Member', ['gross income']].sum().round()[0]
Ph_gR_N_sum = 'Normal Revenue'
PortHarcourt_gR_N_sum = PortHarcourt_branch_dataset.loc[PortHarcourt_branch_dataset['Customer type'] == 'Normal', ['Gross Revenue']].sum().round()[0]
Ph_gInc_N_sum = 'Normal Income'
PortHarcourt_gInc_N_sum = PortHarcourt_branch_dataset.loc[PortHarcourt_branch_dataset['Customer type'] == 'Normal', ['gross income']].sum().round()[0]
Ph_M_T_sum = 'Member {:.0%} Tax'.format(.05)
PortHarcourt_M_T_sum = PortHarcourt_branch_dataset.loc[PortHarcourt_branch_dataset['Customer type'] == 'Member', ['Tax 5%']].sum().round()[0]
Ph_N_T_sum = 'Normal {:.0%} Tax'.format(.05)
PortHarcourt_N_T_sum  = PortHarcourt_branch_dataset.loc[PortHarcourt_branch_dataset['Customer type'] == 'Normal', ['Tax 5%']].sum().round()[0]

####################################################################################################
# 000 - DROPDOWN AND BARCHART
####################################################################################################
Customer_option = ['Store Gross Revenue','Store Quantity Order','Store Gross income']
def dropdow():
     return html.Div([
          dbc.Card(
          dbc.CardBody([
                        dcc.Dropdown(id='customer_type', 
                                        # Update dropdown values using list comphrehension
                                        options=[{
                                            'label': i,
                                            'value': i
                                            } for i in Customer_option ],
                                        placeholder="Select Customer", searchable = True , value = 'Gross Revenue',)
                    ])
                )
     ])

def bar_fig(id):
       return  html.Div([
       dbc.Card(
       dbc.CardBody([
              dcc.Graph(id,
                        style={"height":"36.6vh"})
       ])
       ),  
])


####################################################################################################
# 000 - DROPDOWN AND LINECHAR
####################################################################################################
series_option = ['Product line Time Series','Qauntity Purchase Time Series','Gross Revenue Time Series',]
def dropdow1():
     return html.Div([
          dbc.Card(
          dbc.CardBody([
                        dcc.Dropdown(id='series_type', 
                                        # Update dropdown values using list comphrehension
                                        options=[{
                                            'label': i,
                                            'value': i
                                            } for i in series_option ],
                                        placeholder="Select Series", searchable = True , value = 'Qauntity Purchase Time Series',)
                    ])
                )
     ])

def line_fig(id):
       return  html.Div([
       dbc.Card(
       dbc.CardBody([
              dcc.Graph(id,
                        style={"height":"36.6vh", 'overflow':'hidden'})
       ])
       ),  
])




####################################################################################################
# 000 - TREEMAPCHAR
####################################################################################################
def drawTree(data):
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.treemap(data, 
                            path=['Customer type', 'Product line'], 
                            values='Gross Revenue',
                            color='Gross Revenue',
                            color_continuous_scale='RdBu',
                            title='PRODUCT LINE COUNT'
                    ).update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                        autosize=True,
                      

                    ),
                    config={
                        'displayModeBar': False
                    }, style={"height":"45vh"}
                ) 
            ])
        ),  
    ])



####################################################################################################
# 000 - PIECHARTS
####################################################################################################
Gross_revenue = Abuja_branch_dataset['Gross Revenue']
Gross_income = Abuja_branch_dataset['gross income']
gender = Abuja_branch_dataset['Gender']
Customer_type = Abuja_branch_dataset['Customer type']
RG = 'Revenue by Gender'
IG = 'Income by Gender'
RC = 'Revenue by Customer'
IC = 'Income by Customer'

Gross_revenueL = Lagos_branch_dataset['Gross Revenue']
Gross_incomeL = Lagos_branch_dataset['gross income']
genderL = Lagos_branch_dataset['Gender']
Customer_typeL = Lagos_branch_dataset['Customer type']
RGL = 'Revenue by Gender'
IGL = 'Income by Gender'
RCL = 'Revenue by Customer'
ICL = 'Income by Customer'

Gross_revenueP = PortHarcourt_branch_dataset['Gross Revenue']
Gross_incomeP = PortHarcourt_branch_dataset['gross income']
genderP = PortHarcourt_branch_dataset['Gender']
Customer_typeP = PortHarcourt_branch_dataset['Customer type']
RGP = 'Revenue by Gender'
IGP = 'Income by Gender'
RCP = 'Revenue by Customer'
ICP = 'Income by Customer'


def drawPie(value, name, title):
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.pie(
                        Abuja_branch_dataset, values=value, names=name, title=title, hole = 0.6
                    ).update_layout(
                        template='plotly_dark',
                        plot_bgcolor= '#000000',
                        paper_bgcolor= '#000000',
                        autosize=True,
                        # width=310,
                        # height=450
                    ),
                    config={
                        'displayModeBar': False
                    }, style={"height":"40vh","width":"100%"}
                ) 
            ])
        ),  
    ])


####################################################################################################
# 000 - TEXT
####################################################################################################

color = "danger"


def drawText(name, val):
    return html.Div([
        dbc.Card(
        html.Div(
            [
                html.H3(
                    [
                        html.H5(name),
                    ]
                ),
                html.H6(f"${val:,}"),
            ],
            className=f"border-{color} border-end border-4" f"border-{color} border-start border-5",
        ),
        className="text-center text-nowrap my-2 p-2",
    ),
    ], style={'width':'140px','box-shadow' : '0px 0px 17px 0px rgba(186, 218, 212, .5)'})

def drawText1(name, val):
    return html.Div([
        dbc.Card(
        html.Div(
            [
                html.H3(
                    [
                        html.H5(name),
                    ]
                ),
                html.H6(f"${val:,}" ,style={'color':'#bA6800'}),
            ],
            className=f"border-{color} border-end border-4" f"border-{color} border-start border-5",
            
        ),
        className="text-center text-nowrap my-2 p-2",
    ),
    ], style={'width':'140px','box-shadow' : '0px 0px 17px 0px rgba(186, 218, 212, .5)'})

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
},


import numpy as np
import matplotlib as mpl
colors={}
def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
c1='#FAA831' 
c2='#9A4800' 
n=9
for x in range(n+1):
    colors['level'+ str(n-x+1)] = colorFader(c1,c2,x/n) 
colors['background'] = '#232425'
colors['text'] = '#fff'

agg_Product_City = XYZ_Company_Dataset.groupby(['Product line','City']).agg({"Gross Revenue" : "mean"}).reset_index()
agg_Product_City=agg_Product_City[agg_Product_City['Gross Revenue']>0]
def drawSun_bst():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.sunburst(agg_Product_City, path=['Product line','City'], values='Gross Revenue',
                                       color='Gross Revenue',
                    color_continuous_scale=[colors['level2'], colors['level10']],

                    ).update_layout(
                        title_text='Product line & Cities',
                        paper_bgcolor='#232425',
                        plot_bgcolor='#232425',
                        font=dict(color=colors['text']),
                        height=400
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
            ])
        ),  
    ])






agg_Product_Rev = XYZ_Company_Dataset.groupby('Product line').agg({"Gross Revenue" : "mean"}).reset_index().sort_values(by='Gross Revenue', ascending=False)
agg_Product_Rev['color'] = colors['level10']
agg_Product_Rev['color'][:1] = colors['level1']
agg_Product_Rev['color'][1:2] = colors['level2']
agg_Product_Rev['color'][2:3] = colors['level3']
agg_Product_Rev['color'][3:4] = colors['level4']
agg_Product_Rev['color'][4:5] = colors['level5']
def drawBar_Eng():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=go.Figure(data=[go.Bar(x=agg_Product_Rev['Gross Revenue'],
                                                y=agg_Product_Rev['Product line'], 
                                                marker=dict(color= agg_Product_Rev['color']),
                                                name='Product line', orientation='h',
                                                text=agg_Product_Rev['Gross Revenue'].astype(int),
                                                textposition='auto',
                                                hoverinfo='text',
                                                hovertext=
                                                '<b>Product line</b>:'+ agg_Product_Rev['Product line'] +'<br>' +
                                                '<b>Sales</b>:'+ agg_Product_Rev['Gross Revenue'].astype(int).astype(str) +'<br>' ,
                                                # hovertemplate='Family: %{y}'+'<br>Sales: $%{x:.0f}'
                                                )]

                    ).update_layout(
                        title_text='Best-Selling Products ',
                        paper_bgcolor=colors['background'],
                        plot_bgcolor=colors['background'],
                        font=dict(size=14, color='white'),
                        height=400
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
            ])
        ),  
    ])



avrg_Rev_City = XYZ_Company_Dataset.groupby('City').agg({"Gross Revenue" : "sum"}).reset_index().sort_values(by='Gross Revenue', ascending=False)
def drawPie_Ravrg():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=go.Figure(data=[go.Pie(values=avrg_Rev_City['Gross Revenue'], labels=avrg_Rev_City['City'], name='City',
                                                marker=dict(colors=[colors['level1'],colors['level3'],colors['level5'],colors['level7'],colors['level9']]),
                                                hole=0.7,hoverinfo='label+percent+value', textinfo='label')]

                    ).update_layout(
                        title_text='Revenue Vs City',
                        paper_bgcolor="#000000",
                        plot_bgcolor='#1f2c56',
                        font=dict(size=14,color='white'),
                        height=400
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
            ])
        ),  
    ])



df_city_sa = XYZ_Company_Dataset.groupby('City').agg({"Gross Revenue" : "mean"}).reset_index().sort_values(by='Gross Revenue', ascending=False)
df_city_sa['color'] = colors['level10']
df_city_sa['color'][:1] = colors['level1']
df_city_sa['color'][1:2] = colors['level2']
df_city_sa['color'][2:3] = colors['level3']
df_city_sa['color'][3:4] = colors['level4']
df_city_sa['color'][4:5] = colors['level5']
def drawBar_RavrgC():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=go.Figure(data=[go.Bar(y=df_city_sa['Gross Revenue'],
                             x=df_city_sa['City'], 
                             marker=dict(color= df_city_sa['color']),
                             name='State',
                             text=df_city_sa['Gross Revenue'].astype(int),
                             textposition='auto',
                             hoverinfo='text',
                             hovertext=
                            '<b>City</b>:'+ df_city_sa['City'] +'<br>' +
                            '<b>Gross Revenue/b>:'+ df_city_sa['Gross Revenue'].astype(int).astype(str) +'<br>' ,
                            # hovertemplate='Family: %{y}'+'<br>Sales: $%{x:.0f}'
                            )]
                    ).update_layout(
                        title_text='Average Revenue Vs City',
                        paper_bgcolor=colors['background'],
                        plot_bgcolor=colors['background'],
                        font=dict(size=14,color='white'),
                        height=400
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
            ])
        ),  
    ])


avrg_time_sale = XYZ_Company_Dataset.groupby('Time').agg({"Gross Revenue" : "sum"}).reset_index()
def drawLine_RavrgT():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(style={'overflow':'hidden'},
                    figure=go.Figure(data=[go.Scatter(x=avrg_time_sale['Time'], 
                        y=avrg_time_sale['Gross Revenue'], 
                        fill='tozeroy', fillcolor='#FAA831', 
                        line_color='#bA6800' )]
                    ).update_layout(
                        title_text='Daily Revenue By Time',
                        height=400, width= 3500, paper_bgcolor='#232425',
                        plot_bgcolor='#232425',
                        font=dict(size=14,color='white')
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
            ])
        ),  
    ])


####################################################################################################
# 000 - APP PREDICTION LAYOUT
####################################################################################################
layout = html.Div([

    dbc.Card(
        dbc.CardBody([

            dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
                    ),
            dbc.Row([

                    dbc.Col([
                        html.Div(id='output-datatable', style={'maxHeight': '85vh','maxWidth': '100%','overflow':'scroll'}),
                            ], xs = 12 , sm = 12 , md = 12 , lg = 10 , xl = 10),

                    dbc.Col([
                        html.Div([
                            html.Div(id='output-div', children=[], style={'maxHeight': '85vh','overflow':'scroll'})
                        ], className='row'),

                            ], xs = 12 , sm = 12 , md = 12 , lg = 2 , xl = 2),

                    ])
     
        ]), 
    )
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.P("Select Prediction Model"),
        dcc.Dropdown(id='model_type',
                                    options=["Regression", "RandomForest", "XgbModel" ],
                                    value = 'Regression',
                                    clearable=False),
                                    
        html.Button(id="submit-button", children="Predict Sales"),
        html.Hr(),
        

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            page_size=15
        ),
        dcc.Store(id='stored-data', data=df.to_dict('records')),
       

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])



#  xs = 12 , sm = 12 , md = 12 , lg = 'auto' , xl = 'auto'   

model_choice = {'Regression':LinearRegression,'RandomForest':RandomForestRegressor,'XgbModel':XGBRegressor} 
####################################################################################################
# 000 - APP FIRST TAB LAYOUT
####################################################################################################
layout1 = html.Div([

    dbc.Card(
        dbc.CardBody([

            dbc.Row([
                dbc.Col([
                    html.Div([drawText(Ab_cogs_sum,Abuja_cogs_sum),drawText(Ab_gR_sum,Abuja_gR_sum),drawText(Ab_gInc_sum,Abuja_gInc_sum),drawText(Ab_gR_M_sum,Abuja_gR_M_sum),drawText(Ab_gInc_M_sum,Abuja_gInc_M_sum ),drawText(Ab_gR_N_sum,Abuja_gR_N_sum),drawText(Ab_gInc_N_sum,Abuja_gInc_N_sum),drawText(Ab_M_T_sum,Abuja_M_T_sum),drawText(Ab_N_T_sum,Abuja_N_T_sum),drawText(Ab_N_T_sum,Abuja_N_T_sum)],
                             style={'display':'flex','flex-direction':'row','flex-wrap':'wrap','justify-content':'space-around','margin-bottom':'20px'}),
                ], xs = 12 , sm = 12 , md = 12 , lg = 12 , xl = 12),
            ]),

            dbc.Row([
                dbc.Col([drawTree(Abuja_branch_dataset),
                        ], xs = 12 , sm = 12 , md = 12 , lg = 6 , xl = 6),
                

                dbc.Col([dropdow(),bar_fig(id='plot1')
                        ], xs = 12 , sm = 12 , md = 12 , lg = 6 , xl = 6),
                ]),
            dbc.Row([
                
                dbc.Col([dropdow1(),line_fig(id='plot4'),
                            ], xs = 12 , sm = 12 , md = 12 , lg = 7 , xl = 9),

                dbc.Col([html.Div([drawPie(Gross_revenue,gender,RG),drawPie(Gross_income,gender,IG),drawPie(Gross_revenue,Customer_type,RC),drawPie(Gross_income, Customer_type,IC)], 
                                    style={'display':'flex', 'flex-direction':'column','maxHeight': '48vh', 'overflow': 'scroll'})
                        ], xs = 12 , sm = 12 , md = 12 , lg = 5 , xl = 3),
                ]),


     
        ]),
    )
])






####################################################################################################
# 000 - APP SECOND TAB LAYOUT
####################################################################################################
layout2 = html.Div([

    dbc.Card(
        dbc.CardBody([

            dbc.Row([
                dbc.Col([
                    html.Div([drawText(La_cogs_sum,Lagos_cogs_sum),drawText(La_gR_sum,Lagos_gR_sum),drawText(La_gInc_sum,Lagos_gInc_sum),drawText(La_gR_M_sum,Lagos_gR_M_sum),drawText(La_gInc_M_sum,Lagos_gInc_M_sum ),drawText(La_gR_N_sum,Lagos_gR_N_sum),drawText(La_gInc_N_sum,Lagos_gInc_N_sum),drawText(La_M_T_sum,Lagos_M_T_sum),drawText(La_N_T_sum,Lagos_N_T_sum)],
                             style={'display':'flex','flex-direction':'row','flex-wrap':'wrap','justify-content':'space-around','margin-bottom':'20px'}),
                ], xs = 12 , sm = 12 , md = 12 , lg = 12 , xl = 12),
            ]),

            dbc.Row([
                dbc.Col([drawTree(Lagos_branch_dataset),
                        ], xs = 12 , sm = 12 , md = 12 , lg = 6 , xl = 6),
                

                dbc.Col([dropdow(),bar_fig(id='plot2')
                        ], xs = 12 , sm = 12 , md = 12 , lg = 6 , xl = 6),
                ]),
            dbc.Row([
                
                dbc.Col([dropdow1(),line_fig(id='plot5'),
                            ], xs = 12 , sm = 12 , md = 12 , lg = 7 , xl = 9),

                dbc.Col([html.Div([drawPie(Gross_revenueL,genderL,RGL),drawPie(Gross_incomeL,genderL,IGL),drawPie(Gross_revenueL,Customer_typeL,RCL),drawPie(Gross_incomeL, Customer_typeL,ICL)], 
                                    style={'display':'flex', 'flex-direction':'column','maxHeight': '48vh', 'overflow': 'scroll'})
                        ], xs = 12 , sm = 12 , md = 12 , lg = 5 , xl = 3),
                ]),


     
        ]),
    )
])


####################################################################################################
# 000 - APP THIRD TAB LAYOUT
####################################################################################################
layout3 = html.Div([

    dbc.Card(
        dbc.CardBody([

            dbc.Row([
                dbc.Col([
                    html.Div([drawText(Ph_cogs_sum,PortHarcourt_cogs_sum),drawText(Ph_gR_sum,PortHarcourt_gR_sum),drawText(Ph_gInc_sum,PortHarcourt_gInc_sum),drawText(Ph_gR_M_sum,PortHarcourt_gR_M_sum),drawText(Ph_gInc_M_sum,PortHarcourt_gInc_M_sum ),drawText(Ph_gR_N_sum,PortHarcourt_gR_N_sum),drawText(Ph_gInc_N_sum,PortHarcourt_gInc_N_sum),drawText(Ph_M_T_sum,PortHarcourt_M_T_sum),drawText(Ph_N_T_sum,PortHarcourt_N_T_sum)],
                             style={'display':'flex','flex-direction':'row','flex-wrap':'wrap','justify-content':'space-around','margin-bottom':'20px'}),
                ], xs = 12 , sm = 12 , md = 12 , lg = 12 , xl = 12),
            ]),

            dbc.Row([
                dbc.Col([drawTree(PortHarcourt_branch_dataset),
                        ], xs = 12 , sm = 12 , md = 12 , lg = 6 , xl = 6),
                

                dbc.Col([dropdow(),bar_fig(id='plot3')
                        ], xs = 12 , sm = 12 , md = 12 , lg = 6 , xl = 6),
                ]),
            dbc.Row([
                
                dbc.Col([dropdow1(),line_fig(id='plot6'),
                            ], xs = 12 , sm = 12 , md = 12 , lg = 7 , xl = 9),

                dbc.Col([html.Div([drawPie(Gross_revenueP,genderP,RGP),drawPie(Gross_incomeP,genderP,IGP),drawPie(Gross_revenueP,Customer_typeP,RCP),drawPie(Gross_incomeP, Customer_typeP,ICP)], 
                                    style={'display':'flex', 'flex-direction':'column','maxHeight': '48vh', 'overflow': 'scroll'})
                        ], xs = 12 , sm = 12 , md = 12 , lg = 5 , xl = 3),
                ]),


     
        ]),
    )
])



####################################################################################################
# 000 - FORMATTING INFO
####################################################################################################

####################### Corporate css formatting
corporate_colors = {
    'dark-blue-grey' : 'rgb(62, 64, 76)',
    'medium-blue-grey' : 'rgb(77, 79, 91)',
    'superdark-green' : 'rgb(41, 56, 55)',
    'dark-green' : 'rgb(57, 81, 85)',
    'medium-green' : 'rgb(93, 113, 120)',
    'light-green' : 'rgb(186, 218, 212)',
    'pink-red' : 'rgb(255, 101, 131)',
    'dark-pink-red' : 'rgb(247, 80, 99)',
    'white' : 'rgb(251, 251, 252)',
    'light-grey' : 'rgb(208, 206, 206)',
    'dark':'#1e2130'
}

externalgraph_rowstyling = {
    'margin-left' : '15px',
    'margin-right' : '15px'
}

externalgraph_colstyling = {
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : corporate_colors['dark'],
    'background-color' : corporate_colors['dark'],
    'box-shadow' : '0px 0px 17px 0px rgba(186, 218, 212, .5)',
    'padding-top' : '10px'
}

filterdiv_borderstyling = {
    'border-radius' : '0px 0px 10px 10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : corporate_colors['light-green'],
    'background-color' : corporate_colors['light-green'],
    'box-shadow' : '2px 5px 5px 1px rgba(255, 101, 131, .5)'
    }

navbarcurrentpage = {
    'text-decoration' : 'underline',
    'text-decoration-color' : corporate_colors['pink-red'],
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)'
    }

recapdiv = {
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : 'rgb(251, 251, 252, 0.1)',
    'margin-left' : '15px',
    'margin-right' : '15px',
    'margin-top' : '15px',
    'margin-bottom' : '15px',
    'padding-top' : '5px',
    'padding-bottom' : '5px',
    'background-color' : 'rgb(251, 251, 252, 0.1)'
    }

recapdiv_text = {
    'text-align' : 'left',
    'font-weight' : '350',
    'color' : corporate_colors['white'],
    'font-size' : '1.5rem',
    'letter-spacing' : '0.04em'
    }

####################### Corporate chart formatting

corporate_title = {
    'font' : {
        'size' : 16,
        'color' : corporate_colors['white']}
}

corporate_xaxis = {
    'showgrid' : False,
    'linecolor' : corporate_colors['light-grey'],
    'color' : corporate_colors['light-grey'],
    'tickangle' : 315,
    'titlefont' : {
        'size' : 12,
        'color' : corporate_colors['light-grey']},
    'tickfont' : {
        'size' : 11,
        'color' : corporate_colors['light-grey']},
    'zeroline': False
}

corporate_yaxis = {
    'showgrid' : True,
    'color' : corporate_colors['light-grey'],
    'gridwidth' : 0.5,
    'gridcolor' : corporate_colors['dark-green'],
    'linecolor' : corporate_colors['light-grey'],
    'titlefont' : {
        'size' : 12,
        'color' : corporate_colors['light-grey']},
    'tickfont' : {
        'size' : 11,
        'color' : corporate_colors['light-grey']},
    'zeroline': False
}

corporate_font_family = 'Dosis'

corporate_legend = {
    'orientation' : 'h',
    'yanchor' : 'bottom',
    'y' : 1.01,
    'xanchor' : 'right',
    'x' : 1.05,
	'font' : {'size' : 9, 'color' : corporate_colors['light-grey']}
} # Legend will be on the top right, above the graph, horizontally

corporate_margins = {'l' : 5, 'r' : 5, 't' : 45, 'b' : 15}  # Set top margin to in case there is a legend

corporate_layout = go.Layout(
    font = {'family' : corporate_font_family},
    title = corporate_title,
    title_x = 0.5, # Align chart title to center
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor = 'rgba(0,0,0,0)',
    xaxis = corporate_xaxis,
    yaxis = corporate_yaxis,
    height = 270,
    legend = corporate_legend,
    margin = corporate_margins
    )

####################################################################################################
# 000 - DATA MAPPING
####################################################################################################

#Sales mapping
sales_filepath = 'data/datasource.xlsx'

sales_fields = {
    'date' : 'Date',
    'reporting_group_l1' : 'Country',
    'reporting_group_l2' : 'City',
    'sales' : 'Sales Units',
    'revenues' : 'Revenues',
    'sales target' : 'Sales Targets',
    'rev target' : 'Rev Targets',
    'num clients' : 'nClients'
    }
sales_formats = {
    sales_fields['date'] : '%d/%m/%Y'
}

####################################################################################################
# 000 - IMPORT DATA
####################################################################################################


################################################################################################################################################## SET UP END

####################################################################################################
# 000 - DEFINE REUSABLE COMPONENTS AS FUNCTIONS
####################################################################################################


####################################################################################################
# 001 - SALES
####################################################################################################

Decision = html.Div([

    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                     html.Div([drawText1(XYZ_cogs_sum,XYZ_Company_cogs_sum),drawText1(XYZ_gR_sum,XYZ_Company_gR_sum),drawText1(XYZ_gInc_sum,XYZ_Company_gInc_sum),drawText1(XYZ_gR_M_sum,XYZ_Company_gR_M_sum),drawText1(XYZ_gInc_M_sum,XYZ_Company_gInc_M_sum ),drawText1(XYZ_gR_N_sum,XYZ_Company_gR_N_sum),drawText1(XYZ_gInc_N_sum,XYZ_Company_gInc_N_sum),drawText1(XYZ_M_T_sum,XYZ_Company_M_T_sum),drawText1(XYZ_N_T_sum,XYZ_Company_N_T_sum)],
                              style={'display':'flex','flex-direction':'row','flex-wrap':'wrap','justify-content':'space-around','margin-bottom':'20px'}),
                ], xs = 12 , sm = 12 , md = 12 , lg = 12 , xl = 12),
            ]),

            dbc.Row([
                dbc.Col([
                    drawBar_RavrgC()
                ],xs = 12 , sm = 12 , md = 12 , lg = 4 , xl = 4),
                dbc.Col([
                    drawPie_Ravrg()
                ],xs = 12 , sm = 12 , md = 12 , lg = 4 , xl = 4),
                dbc.Col([
                    drawBar_Eng()
                ],xs = 12 , sm = 12 , md = 12 , lg = 4 , xl = 4),
            ]),

            dbc.Row([
                dbc.Col([
                    drawLine_RavrgT()
                ],xs = 12 , sm = 12 , md = 12 , lg = 6 , xl = 6),
                dbc.Col([
                    drawSun_bst()
                ],xs = 12 , sm = 12 , md = 12 , lg = 6 , xl = 6),
            ])


     
        ]),
    )
])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE],
                suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
                )
server = app.server
# app = Dash(external_stylesheets=[dbc.themes.SLATE])
app.layout = html.Div([

    dbc.Card(
        dbc.CardBody([
            build_banner(),

            dbc.Row([
                 dbc.Col([
                     tabHead
        #             dbc.NavLink([
        #             dcc.Link(page['name']+"  |  ", href=page['path'])
        #             for page in dash.page_registry.values()
        # ],),
                 ], width=12)

            ], align='center'), 


     
        ]), color = '#000000', style={'margin-bottom':'-15px'}
    ),
    html.Hr(),

    # content of each page
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    # dash.page_container
    
])
"/Abuja_Branch_Control_Charts"
####################################################################################################
# 000 - TAB CALLBACK
####################################################################################################
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return layout
    elif pathname == "/Abuja_Branch_Control_Charts":
        return layout1
    elif pathname == "/Lagos_Branch_Control_Charts":
        return layout2
    elif pathname == "/Portharcourt_Branch_Control_Charts":
        return layout3
    elif pathname == "/Management_Decision_Engine":
        return Decision
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    ),

####################################################################################################
# 000 - BARCHART CALLBACK
####################################################################################################
@app.callback(Output('plot1', 'figure'),
              [Input('customer_type', 'value')])

def get_graph(Customer_type):
    if Customer_type == "Store Gross Revenue":

        bar_fig = px.bar(Abuja_branch_dataset, 
                 x='Product line', 
                 y='Gross Revenue', color='City',
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    height=350,
                
                    
                )

    elif Customer_type == "Store Quantity Order":
        
        bar_fig = px.bar(Abuja_branch_dataset, 
                 x='Product line', 
                 y='Quantity', color='City',
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    height=350
                )

    else:

        bar_fig = px.bar(Abuja_branch_dataset, 
                 x='Product line', 
                 y='gross income', color='City',
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    height=350
           
                
                )
    
    return bar_fig


@app.callback(Output('plot2', 'figure'),
              [Input('customer_type', 'value')])

def get_graph(Customer_type):
    if Customer_type == "Store Gross Revenue":

        bar_fig = px.bar(Lagos_branch_dataset, 
                 x='Product line', 
                 y='Gross Revenue', color='City',
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    height=350,
                
                    
                )

    elif Customer_type == "Store Quantity Order":
        
        bar_fig = px.bar(Lagos_branch_dataset, 
                 x='Product line', 
                 y='Quantity', color='City',
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    height=350
                )

    else:

        bar_fig = px.bar(Lagos_branch_dataset, 
                 x='Product line', 
                 y='gross income', color='City',
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    height=350
           
                
                )
    
    return bar_fig


@app.callback(Output('plot3', 'figure'),
              [Input('customer_type', 'value')])

def get_graphPh(Customer_type):
    if Customer_type == "Store Gross Revenue":

        bar_fig = px.bar(PortHarcourt_branch_dataset, 
                 x='Product line', 
                 y='Gross Revenue', color='City',
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    height=350,
                
                    
                )

    elif Customer_type == "Store Quantity Order":
        
        bar_fig = px.bar(PortHarcourt_branch_dataset, 
                 x='Product line', 
                 y='Quantity', color='City',
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    height=350
                )

    else:

        bar_fig = px.bar(PortHarcourt_branch_dataset, 
                 x='Product line', 
                 y='gross income', color='City',
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    height=350
           
                
                )
    
    return bar_fig


####################################################################################################
# 000 - LINECHART CALLBACK
####################################################################################################
@app.callback(Output('plot4', 'figure'),
              [Input('series_type', 'value')])

def get_graphline(series_type):
    if series_type == "Product line Time Series":
        
        Abuja_TQ_sort = Abuja_branch_dataset[['Time','Product line']].sort_values(by=['Time'])
        line_fig = px.line(Abuja_TQ_sort, x=Abuja_TQ_sort['Time'], 
                   y=Abuja_TQ_sort['Product line'], 
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    width=3500,
            
                )
    elif series_type == "Qauntity Purchase Time Series":

        Abuja_TQ_sort = Abuja_branch_dataset[['Time','Quantity']].sort_values(by=['Time'])
        line_fig = px.line(Abuja_TQ_sort, x=Abuja_TQ_sort['Time'], 
                   y=Abuja_TQ_sort['Quantity'], 
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    width=3500,
                )
        
    else:
        
        Abuja_TQ_sort = Abuja_branch_dataset[['Time','Gross Revenue']].sort_values(by=['Time'])
        line_fig = px.line(Abuja_TQ_sort, x=Abuja_TQ_sort['Time'], 
                   y=Abuja_TQ_sort['Gross Revenue'], 
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    width=3500,
        
                )
              

    return line_fig

@app.callback(Output('plot5', 'figure'),
              [Input('series_type', 'value')])

def get_graphline(series_type):
    if series_type == "Product line Time Series":
        
        Lagos_TQ_sort = Lagos_branch_dataset[['Time','Product line']].sort_values(by=['Time'])
        line_fig = px.line(Lagos_TQ_sort, x=Lagos_TQ_sort['Time'], 
                   y=Lagos_TQ_sort['Product line'], 
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    width=3500,
            
                ) 
    elif series_type == "Qauntity Purchase Time Series":

        Lagos_TQ_sort = Lagos_branch_dataset[['Time','Quantity']].sort_values(by=['Time'])
        line_fig = px.line(Lagos_TQ_sort, x=Lagos_TQ_sort['Time'], 
                   y=Lagos_TQ_sort['Quantity'], 
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    width=3500,
            
                ) 

    else:
        
        Lagos_TQ_sort = Lagos_branch_dataset[['Time','Gross Revenue']].sort_values(by=['Time'])
        line_fig = px.line(Lagos_TQ_sort, x=Lagos_TQ_sort['Time'], 
                   y=Lagos_TQ_sort['Gross Revenue'], 
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    width=3500,
        
                )
    
    return line_fig

@app.callback(Output('plot6', 'figure'),
              [Input('series_type', 'value')])

def get_graphline(series_type):
    if series_type == "Product line Time Series":
        
        PortHarcourt_TQ_sort = PortHarcourt_branch_dataset[['Time','Product line']].sort_values(by=['Time'])
        line_fig = px.line(PortHarcourt_TQ_sort, x=PortHarcourt_TQ_sort['Time'], 
                   y=PortHarcourt_TQ_sort['Product line'], 
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    width=3500,
            
                )  
    
    elif series_type == "Qauntity Purchase Time Series":

        PortHarcourt_TQ_sort = PortHarcourt_branch_dataset[['Time','Quantity']].sort_values(by=['Time'])
        line_fig = px.line(PortHarcourt_TQ_sort, x=PortHarcourt_TQ_sort['Time'], 
                   y=PortHarcourt_TQ_sort['Quantity'], 
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    width=3500,
            
                )   

    else:

        PortHarcourt_TQ_sort = PortHarcourt_branch_dataset[['Time','Gross Revenue']].sort_values(by=['Time'])
        line_fig = px.line(PortHarcourt_TQ_sort, x=PortHarcourt_TQ_sort['Time'], 
                   y=PortHarcourt_TQ_sort['Gross Revenue'], 
                   ).update_layout(
                    template='plotly_dark',
                    plot_bgcolor= 'rgba(0, 0, 0, 0)',
                    paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    width=3500,
        
                )
    
    return line_fig


@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

 
@app.callback(Output('output-div', 'children'),
              Input('submit-button', 'n_clicks'),
              State('stored-data','data'),
              State('model_type','value'),)
def make_output(n_clicks, dataset, model_name):
    dff = pd.DataFrame(dataset)

    if n_clicks is None:
        return dash.no_update
    
    else:
        data_train = pd.read_csv("data_train.csv")
        target = data_train['Item_Outlet_Sales']
        target = pd.DataFrame(target, columns=['Item_Outlet_Sales'])
        feature = data_train.drop(['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'],axis =1)

        X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size = 0.2, random_state = 42 )
        model = model_choice[model_name]()
        model.fit(X_train, y_train)
        result_prediction = model.predict(dff)
        result_prediction_df = pd.DataFrame(result_prediction, columns=['SALES PREDICTED']) 
        my_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in result_prediction_df.columns],
        data=result_prediction_df.to_dict('records')
    )

        print(result_prediction_df)
        
        return my_table






if __name__ == "__main__":
    app.run(debug=False)

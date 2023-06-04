
# Dash package
from pandas.io.sql import has_table
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_html_components as html
import dash_table
from plotly.subplots import make_subplots

# standard package
import pandas as pd
import numpy as np

#
import re
import os

# env mngt
from dotenv import load_dotenv 
load_dotenv()


# date & time package
import datetime
import math
from datetime import timedelta
#from datetime import datetime


#db package
from sqlalchemy import create_engine
import psycopg2

# local package
from app import app

today = datetime.date.today()
print("Today's date on sage page:", today)


#get env db
host=os.environ.get('HOST')
user=os.environ.get('USER')
database=os.environ.get('DATABASE')
pwd=os.environ.get('PASSWORD')


db_url = 'postgresql://'+user+':'+pwd+'@'+host+'/'+database 
#db_url = os.environ.get('URL')

# Create an engine instance
alchemyEngine = create_engine(db_url)

# Connect to PostgreSQL server
dbConnection = alchemyEngine.connect()
# Read data from PostgreSQL database table and load into a DataFrame instance
df_balance_sage  = pd.read_sql("SELECT * FROM balance_sage", dbConnection)
df_apporteur  = pd.read_sql("SELECT * FROM apporteur", dbConnection)
df_recouvrement_sage  = pd.read_sql("SELECT * FROM recouvrement_sage", dbConnection)
df_recouvrement_garantie  = pd.read_sql("SELECT * FROM recouvrement_garantie", dbConnection)

#gestion de df
df = df_balance_sage.join(df_apporteur.set_index('N°ASSURE'), on = 'N°ASSURE')
df ['APPORTEUR'] = df ['APPORTEUR'].fillna('-')

#gestion df garantie
df_recouvrement_garantie['date'] = [x.date() for x in df_recouvrement_garantie['date']]


#print(df.head(5))
#print(df.columns)


#date reference
dt_range = df['date_extraction'].unique()
dt_range = list(dt_range)
dt_range.sort(reverse = True)
dt_ref = dt_range[0]
dt_var = dt_range[1]

print('Last date in sage db: ',dt_ref)
import sys
print("mem sage: ", sys.getsizeof(df))
print()

list_numClient = df['N°ASSURE'].unique()
#print(list_numClient[3080:4080] )
list_numClient = list_numClient[list_numClient != np.array(None)]
#list_numClient = list_numClient[~np.isnull(list_numClient)]
#a[a != np.array(None)]
#exit()
list_numClient.sort()


list_apporteur = df ['APPORTEUR'].unique()
list_apporteur = list(list_apporteur)
list_apporteur.sort()
list_apporteur.insert(0,'TOTAL')
list_apporteur

affichage_list = dict()
affichage_list = {
            'solde': ['Solde', 'solde'],
            'credit' : ['Crédit','CREDIT'],
            'debit' :['Débit','DEBIT']
}

list_SageSituationRecouvrement = [
    'Liste-fev2020-NH',
    'recouvrable',
    'irrecouvrable'
]


#les sous-df
df_soldeTotal = df.groupby(['date_extraction'])['solde'].sum()
df_soldeTotal = pd.DataFrame(df_soldeTotal.reset_index())

df_soldeApporteur = df.groupby(['APPORTEUR', 'date_extraction'])['solde'].sum()
df_soldeApporteur = pd.DataFrame(df_soldeApporteur.reset_index())

df_recouvrement_garantie1 = df_recouvrement_garantie.groupby(['date'])['montant'].sum()
df_recouvrement_garantie1 = pd.DataFrame(df_recouvrement_garantie1.reset_index())


df_recouvrement_garantie1b = df_recouvrement_garantie.join(df_apporteur.set_index('N°ASSURE'), on = 'N°ASSURE')
df_recouvrement_garantie1b = df_recouvrement_garantie1b.groupby(['APPORTEUR','date'])['montant'].sum()
df_recouvrement_garantie1b = pd.DataFrame(df_recouvrement_garantie1b.reset_index())



def portfolio_dynamic_toDate(b,dt,dt_var):
    x = portfolio_dynamic(b)
    x1 = x.loc[dt,:][0]
    v = x.loc[dt_var:dt,'Variation'].sum()
    #dv = x.index[len(x)-2]
    #xVar = x.loc[dt_var,:][0]
    #v = x1-xVar
    diff_date = np.busday_count( dt_var, dt, weekmask =[1,1,1,1,1,0,0] ) + 1
    #moy = v / diff_date.days
    moy = v / diff_date
    return x1, v, diff_date, moy

def portfolio_dynamic(b):
    
    if b == 'solde': 
        x = df.groupby('date_extraction')[affichage_list[b][1]].sum()
    elif b == 'credit':
        x =  df.groupby('date_extraction')[affichage_list[b][1]].sum()
    elif b == 'debit':
        x = df.groupby('date_extraction')[affichage_list[b][1]].sum()
    
    x = pd.DataFrame(x)
    x = x.join(x.shift(),rsuffix='_1')
    x['Variation'] = x.iloc[:,0]-x.iloc[:,1]
    x.fillna(0, inplace=True)
    return x

def card_content(b,dt,dt_var):

    res = [
        dbc.CardHeader(affichage_list[b][0]),
        dbc.CardBody(
            [
                html.H5('{:,.0f}'.format(portfolio_dynamic_toDate(b,dt,dt_var)[0]), className="card-title"),
                html.P(str('{:,.0f}'.format(portfolio_dynamic_toDate(b,dt,dt_var)[1]))+" (Variation sur " + str(portfolio_dynamic_toDate(b,dt,dt_var)[2]) + " jours)", className="card-text"),
                html.P(str('{:,.1f}'.format(portfolio_dynamic_toDate(b,dt,dt_var)[3],1))+" (moyenne par jour)" ),
                
                
            ]
        ),
    ]

    return res

def portfolio_dynamic_graph(b):
    
    fig = go.Figure()

    dynamique1 = portfolio_dynamic(b)
    fig.add_trace(go.Bar(x=dynamique1.index,
                        y=dynamique1[affichage_list[b][1]],
                        marker_color='indianred'
                        )
    )

    fig.update_layout( title_text= affichage_list[b][0])
    fig.update_layout(title_x=0.5)
    fig.update_layout(xaxis_range=[datetime.datetime(2021, 12, 1),
                               datetime.datetime(2023, 12, 31)])

    dcc_graph = dcc.Graph(figure = fig)

    return dcc_graph

def xdcc_table (dfx):
    
    dcc_table = dash_table.DataTable(
        #id='table',
        columns=[{"name": i, "id": i} for i in dfx.columns],
        data=dfx.to_dict('records'),
        style_table={
            'maxHeight': 500,
            'overflowY': 'scroll',
            'overflowX': 'scroll',
            #'width': '100%',
            #'minWidth': '100%'
        },
        style_header={
            'fontWeight': 'bold',
            'backgroundColor': 'rgb(105,105,105)',
            'color': 'white',
            'textAlign': 'center',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(225, 225, 240)'
            },
        ]


    )
    return dcc_table


#gestion df par client
def df_byClient(dt):
    dfx = df[df['date_extraction']==dt]
    dfx1 = dfx.groupby('N°ASSURE')['solde'].sum() 
    dfx1 = pd.DataFrame(dfx1)
    dfx1 = dfx1.sort_values('solde', ascending=True)
    dfx1 = dfx1.join(df_apporteur.set_index('N°ASSURE'))
    dfx1 = dfx1[['NOM','APPORTEUR','solde']]
    dfx1.reset_index(inplace=True)
    dfx1b = dfx1.copy()
    dfx1b['solde'] = dfx1b['solde'].map("{:,.0f}".format)
    
    res = dict()
    res ={ 'value': dfx1,
           'formated': dfx1b,
    }
    
    return res

def df_byClient_var(dt, dt_var):

    dfx = df_byClient(dt)['value']
    dfx_var = df_byClient(dt_var)['value']
    dfx_var2 = dfx_var [['N°ASSURE','solde']].copy()
    #dfx_var2 = pd.DataFrame(dfx_var2).copy()
    dfx_var2.columns = ['N°ASSURE','solde_n']
    dfy = dfx.join(dfx_var2.set_index('N°ASSURE'), on = 'N°ASSURE')
    dfy['solde_var'] =  dfy['solde'] - dfy['solde_n']
    
    #sorted by soldeMove
    dfy1 = dfy.sort_values('solde_var', ascending=False)
    dfy_creditMv = dfy1.head(10)
    dfy_debitMv = dfy1.tail(10)

    
    dfy1b = dfy1.copy()
    dfy1b['solde'] = dfy1b['solde'].map("{:,.0f}".format)
    dfy1b['solde_n'] = dfy1b['solde_n'].map("{:,.0f}".format)
    dfy1b['solde_var'] = dfy1b['solde_var'].map("{:,.0f}".format)

    dfy_creditMv_f = dfy_creditMv.copy()
    dfy_creditMv_f['solde'] = dfy_creditMv_f['solde'].map("{:,.0f}".format)
    dfy_creditMv_f['solde_n'] = dfy_creditMv_f['solde_n'].map("{:,.0f}".format)
    dfy_creditMv_f['solde_var'] = dfy_creditMv_f['solde_var'].map("{:,.0f}".format)

    dfy_debitMv_f = dfy_debitMv.copy()
    dfy_debitMv_f['solde'] = dfy_debitMv_f['solde'].map("{:,.0f}".format)
    dfy_debitMv_f['solde_n'] = dfy_debitMv_f['solde_n'].map("{:,.0f}".format)
    dfy_debitMv_f['solde_var'] = dfy_debitMv_f['solde_var'].map("{:,.0f}".format)
    
    #sorted by level of the current solde
    dfy2 = dfy.sort_values('solde', ascending=True)
    dfy2b = dfy2.copy()
    dfy2b['solde'] = dfy2b['solde'].map("{:,.0f}".format)
    dfy2b['solde_n'] = dfy2b['solde_n'].map("{:,.0f}".format)
    dfy2b['solde_var'] = dfy2b['solde_var'].map("{:,.0f}".format)
    
    #df joined with recouvrement & garantie
    df_recouvrement_garantie2 = df_recouvrement_garantie[df_recouvrement_garantie['date']> dt]
    df_recouvrement_garantie2 = df_recouvrement_garantie2.groupby('N°ASSURE')['montant'].sum()
    df_recouvrement_garantie2 = pd.DataFrame(df_recouvrement_garantie2)
    df_recouvrement_garantie2.columns = ['garantie']
    
    dfy3 =  dfy2 .join(df_recouvrement_sage.set_index("N°CLIENT"), on = 'N°ASSURE')
    dfy3 =  dfy3 .join(df_recouvrement_garantie2, on = 'N°ASSURE')
    dfy3 = dfy3 [['N°ASSURE', 'NOM', 'APPORTEUR', 'solde', 'solde_n', 'solde_var','garantie',
      'situation','commentaires']]
    dfy3b = dfy3.copy()
    dfy3b['solde'] = dfy3b['solde'].map("{:,.0f}".format)
    dfy3b['solde_n'] = dfy3b['solde_n'].map("{:,.0f}".format)
    dfy3b['solde_var'] = dfy3b['solde_var'].map("{:,.0f}".format)
    dfy3b['garantie'] = dfy3b['garantie'].map("{:,.0f}".format)

    
    
    res = dict()
    res ={ 'value_bySoldeMov': dfy1,
           'value_bySoldeMov_formated': dfy1b,
           'creditMv' :dfy_creditMv_f,
           'debitMv': dfy_debitMv_f,
           'value_bySoldeCur': dfy2,
          'value_bySoldeCur_formated': dfy2b,
         'value_bySoldeCur_rec': dfy3,
          'value_bySoldeCur_rec_formated': dfy3b,
          
          
    }
    
    return res

def debiteur_resume(dt, dtvar):

    dfx = df_byClient_var(dt, dtvar)['value_bySoldeCur_rec']
    dfx['APPORTEUR']  = dfx['APPORTEUR'] .fillna('-')
    dfx['situation']  = dfx['situation'] .fillna('-')


    cross = pd.crosstab(dfx['APPORTEUR'], dfx['situation'],values =dfx['solde'],aggfunc = sum ,
                        margins = True, margins_name = 'Total')
    cross.reset_index(inplace = True)
      
    res1 = cross.copy()
    res1 = res1.sort_values('Total', ascending=True)
    res1['-'] = res1['-'].map("{:,.0f}".format)
    res1['Liste-fev2020-NH'] = res1['Liste-fev2020-NH'].map("{:,.0f}".format)
    res1['recouvrable'] = res1['recouvrable'].map("{:,.0f}".format)
    res1['Total'] = res1['Total'].map("{:,.0f}".format)
    
    
    cross_var = pd.crosstab(dfx['APPORTEUR'], dfx['situation'],values =dfx['solde_var'],aggfunc = sum ,
                        margins = True, margins_name = 'Total')
    cross_var.reset_index(inplace = True)
    res2 = cross_var.copy()
    res2 = res2.sort_values('Total', ascending=False)
    res2['-'] = res2['-'].map("{:,.0f}".format)
    res2['Liste-fev2020-NH'] = res2['Liste-fev2020-NH'].map("{:,.0f}".format)
    res2['recouvrable'] = res2['recouvrable'].map("{:,.0f}".format)
    res2['Total'] = res2['Total'].map("{:,.0f}".format)
   

    res = dict()
    res = { 'value' : cross,
            'value_formated' : res1,
           'variation':cross_var,
           'variation_formated':res2,
           

    }

    return res

def recouvrement_filter (dfx, col, val):
    res = dfx[dfx[col]==val]
    return res

# solde par client
def sageClientSolde_table(dt,c):

    dfx = df[df['date_extraction']== dt]
    dfx = dfx[dfx['N°ASSURE']== c]

    dfx = dfx[['N°ASSURE', 'NOM','PIECE', 'N°POLICE', 'LIBELLE', 'OPE', 'EFFET', 'DEBIT', 'CREDIT']]
    solde = dfx['CREDIT'].sum() - dfx['DEBIT'].sum()
    
    dfx['DEBIT'] = dfx['DEBIT'].map("{:,.0f}".format)
    dfx['CREDIT'] = dfx['CREDIT'].map("{:,.0f}".format)
    
    res = dict()
    res = { 'df' : dfx,
            'solde' : solde
    }

    return res


#graphique solde par apporteur
def solde_graph(apporteur):
    
    if apporteur == 'TOTAL':
        df_graph = df_soldeTotal.copy()
        df_graph2 = df_recouvrement_garantie1.copy()
    else: 
        df_graph = df_soldeApporteur[df_soldeApporteur['APPORTEUR']== apporteur]  
        df_graph2 = df_recouvrement_garantie1b[df_recouvrement_garantie1b['APPORTEUR']== apporteur]  
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_graph['date_extraction'],
                         y=df_graph['solde'],
                         name="SOLDE",
                         marker_color='indianred'
                        )
    )
    fig.add_trace(go.Bar(x=df_graph2['date'],
                         y=df_graph2['montant'],
                         name="GARANTIE",
                         marker_color='green'
                        )
    )

    fig.update_layout( title_text= apporteur)
    fig.update_layout(title_x=0.5)
    fig.update_layout(xaxis_range=[datetime.datetime(2021, 12, 1),
                               datetime.datetime(2023, 12, 31)])

    #fig.update_layout(yaxis_range=[df_graph['solde'].min()*1.05,0])

    dcc_graph = dcc.Graph(figure = fig)


    return dcc_graph


def recouvrement_select(numAss, dt, dtvar):
    
    dfx = df_byClient_var(dt,dtvar)['value_bySoldeCur_rec_formated']
    dfx = dfx[dfx['N°ASSURE']==numAss]
    
    app = list(dfx['APPORTEUR'])[0]
    nom = list(dfx['NOM'])[0]
    sol = list(dfx['solde'])[0]
    com = list(dfx['commentaires'])[0]
    sit = list(dfx['situation'])[0]

    res = dict()
    res = { 
        'apporteur':app,
        'nom':nom,
        'solde':sol,
        'commentaire': com,
        'situation':sit,
    }

    return res


layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([                  
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody(
                            
                            [
                                html.P('Référence'),
                                dcc.Dropdown(
                                    id='dd-dt-ref-sage',
                                    options=[
                                        {'label': i, 'value': i} for i in dt_range
                                    ],
                                    value=dt_ref,
                                    clearable = False,
                                    style={
                                        'font-size': "90%",
                                        },
                                ),
                                
                            ]
                        ),

                    ],style={'text-align': 'center'}),
                    
                ]),

                html.Br(),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody(
                            [
                               html.P('Changement'),
                               dcc.Dropdown(
                                    id='dd-dt-var-sage',
                                    options=[
                                        {'label': i, 'value': i} for i in dt_range
                                    ],
                                    value=dt_var,
                                    clearable = False,
                                    style={
                                        'font-size': "90%",
                                        },
                                ),

                        
                            ]
                        )
                    ],style={'text-align': 'center'}) 
                    
                ]),
                html.Br(),
              
                

              


            ],xs=12, sm=6, md=6, lg=2, xl=2),

            dbc.Col([
                dbc.Tabs(
                [
                    dbc.Tab(label="Balance-Globale", tab_id="globale-id-sage"),
                    dbc.Tab(label="Balance-Clients", tab_id="client-id-sage"),
                    dbc.Tab(label="Suivi-Débiteur", tab_id="debiteur-id-sage"),
                    
                ],
                id="tabs",
                active_tab="None",
                ),
                html.Br(),
                html.Div(id="tab-content-sage")#, className="p-4"),
            ],xs=12, sm=6, md=6, lg=10, xl=10
            ),

            
        ]),
    ],fluid=True
)


#callback tab
@app.callback(
    Output("tab-content-sage", "children"),
    Input("tabs", "active_tab"),
)
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab is not None:
        
        if active_tab == "globale-id-sage":
            
            chld = [
                # #html.P(),
                # dbc.Row(dbc.Col('En chiffres', className='h4')),
                # dbc.Row(
                #     [
                #         dbc.Col([dbc.Card(
                #             (card_content('solde',dt_ref,dt_var)),style={'text-align': 'center'}, color="danger", outline=True, id='card-souscripteur-id')
                #         ],xs=12, sm=12, md=12, lg=4, xl=4),
                #         dbc.Col([dbc.Card(
                #             (card_content('credit',dt_ref,dt_var)),style={'text-align': 'center'}, color="danger", outline=True,id='card-police-id')
                #         ],xs=12, sm=12, md=12, lg=4, xl=4),
                #         dbc.Col([dbc.Card(
                #             (card_content('debit',dt_ref,dt_var)),style={'text-align': 'center'}, color="danger", outline=True,id='card-quittance-id')
                #         ],xs=12, sm=12, md=12, lg=4, xl=4),
                #     ],
                # ),
                # html.Br(),

                # dbc.Row(dbc.Col('Evolution -  Convergence vers 0', className='h4')),
                # dbc.Row(
                #    [
                #         dbc.Col([portfolio_dynamic_graph("solde"), 
                #         ], xs=12, sm=12, md=12, lg=12, xl=12),
                #     ],
                # ),
                html.Hr(),
                dbc.Row(dbc.Col('Liste des variations', className='h4')),
                dbc.Row(dbc.Col('Au CREDIT')),
                dbc.Row(
                    dbc.Col(
                        id = 'liste-id-sage-credit',
                        xs=12, sm=12, md=12, lg=12, xl=12)
                ),
                html.Hr(),
                dbc.Row(dbc.Col('Au DEBIT')),
                dbc.Row(
                    dbc.Col(
                        id = 'liste-id-sage-debit',
                        xs=12, sm=12, md=12, lg=12, xl=12)
                )
                
            ]
            
            
            return  chld


        elif active_tab == "client-id-sage":   
            chld =  [
                dbc.Row([    
                    
                    dbc.Col([
                        dcc.Dropdown(
                            id='dd-numClient-id',
                            options=[
                                {'label': i, 'value': i} for i in list_numClient
                            ],
                            optionHeight=45,
                            #value=list_souscripteur[0],
                            clearable = False,
                            style={
                                'font-size': "90%",
                            },
                        )
                    ], xs=12, sm=12, md=12, lg=4, xl=4)
                ]),
                html.Br(),
                dbc.Row(
                    dbc.Col(
                        #children = souscripteur_graph(dt_ref,list_souscripteur[0]),
                        id = 'SageClient-graph-id',
                        xs=12, sm=12, md=12, lg=12, xl=12  
                    ),
                ),
                html.Br(),
                dbc.Row(
                    dbc.Col(
                        #souscripteur_table(dt_ref,list_souscripteur[0]),
                        id = 'SageClient-table-id',
                        xs=12, sm=12, md=12, lg=12, xl=12
                    )   
                )
            ]
            return  chld


        elif active_tab == "debiteur-id-sage":     
            chld =  [
                dbc.Row(dbc.Col('Résumé en chiffre', className='h5')),
                dbc.Row(
                    dbc.Col([
                        dcc.RadioItems(
                            id = 'radio-valVar-id',
                            options=[
                                    {'label': i, 'value': i} for i in ['valeur', 'variation']
                            ],
                            value='valeur',
                            labelStyle={'display': 'inline-block'},
                            inputStyle={"margin-left": "10px"},
                            
                        ),

                    ], xs=12, sm=12, md=12, lg=4, xl=4),
                    
                ),
                dbc.Row(
                    dbc.Col(
                        id = 'sageDebiteur-resume-id',
                        xs=12, sm=12, md=12, lg=12, xl=12
                    ),
                ),
   
                html.Br(),
                dbc.Row(dbc.Col('Evolution par apporteur', className='h5')),
                dbc.Row(
                    dbc.Col([
                        dcc.Dropdown(
                            id='dd-apporteur-id',
                            options=[
                                {'label': i, 'value': i} for i in list_apporteur
                            ],
                            value="TOTAL",
                            clearable = False,
                        )
                    ], xs=12, sm=12, md=12, lg=4, xl=4),
                    
                ),
                dbc.Row(
                    dbc.Col(
                        id = 'apporteur-graph-id'
                    , xs=12, sm=12, md=12, lg=12, xl=12),
                    
                ),

                
                #html.Br(),
                

                html.Br(),
                dbc.Row(dbc.Col('Liste complète', className='h5')),
                dbc.Row(
                    dbc.Col(
                        id = 'sageDebiteur-table-id',
                        xs=12, sm=12, md=12, lg=12, xl=12
                    )   
                ),

                html.Hr(),
                dbc.Row(dbc.Col('Saisie pour mise à jour', className='h5')),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Col([
                                    dcc.RadioItems(
                                        id = 'radio-sageRecouvrement-id',
                                        options=[
                                                {'label': i, 'value': i} for i in ['Nouvelle saisie', 'Mise à jour']
                                        ],
                                        value='Nouvelle saisie',
                                        labelStyle={'display': 'inline-block'},
                                        inputStyle={"margin-left": "10px"},
                                        style={'font-size': "90%"},  
                                    ),
                                ], 
                                #xs=12, sm=12, md=12, lg=3, xl=3
                                ),

                                dbc.Col([
                                    dbc.Button("RESET", 
                                        id="button-sageRecouvrement", 
                                        color="danger",  block=True, outline=True, size="sm"),
                                ],
                                #xs=12, sm=12, md=12, lg=3, xl=3
                                ),
                                
                            ])

                        ], style = {'text-align': 'center'}),
                        
                    ],xs=12, sm=12, md=12, lg=6, xl=6),
                    
                   dbc.Col([         
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6('Num Assuré'),
                                        dcc.Dropdown(
                                            id='dd-numClient-sageRecouvrement-id',
                                            options=[
                                                {'label': i, 'value': i} for i in list_numClient
                                            ],
                                            #value=list_souscripteur[0],
                                            clearable = False,
                                        )
                                    ], xs=12, sm=12, md=12, lg=6, xl=6),

                                    dbc.Col([
                                            html.H6('Apporteur'),
                                            html.P(
                                                id='input-apporteur-sageRecouvrement-id',
                                                style={'font-size': "90%",'color': 'grey'}
                                            ),
                                    ],
                                    xs=12, sm=12, md=12, lg=6, xl=6
                                    ),

                                ]),
                                #html.Br()
                                

                            ])
                        )

                    ],xs=12, sm=12, md=12, lg=6, xl=6),

                    



                ]),

                
                html.Br(),

                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Nom'),
                                html.P(
                                    id='input-nom-sageRecouvrement-id',
                                    style={'font-size': "90%",'color': 'grey'}
                                ),  
                                
                            ], xs=12, sm=12, md=12, lg=4, xl=4),
                                
                            dbc.Col([
                                html.H6('Solde'),  
                                html.P(
                                    id='input-solde-sageRecouvrement-id',
                                    style={'font-size': "90%",'color': 'grey'},
                                )

                            ], xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([
                                html.H6('Prime impayée enregistrée'),  
                                dcc.Dropdown(
                                            id='dd-situation-sageRecouvrement-id',
                                            options=[
                                                {'label': i, 'value': i} for i in list_SageSituationRecouvrement
                                            ],
                                            #value=list_souscripteur[0],
                                            clearable = False,
                                        )  
                            ],xs=12, sm=12, md=12, lg=4, xl=4),
           
                        ]),

                    ])
                ),
                
             
                html.Br(),
                dbc.Row(
                    dbc.Col([
                         html.H6('Commentaires - Actions'),
                         dcc.Textarea(
                            id = 'input-commentaire-sageRecouvrement-id',
                            placeholder='',
                            #value=X1_0,
                            style={
                                'width': '100%', 'height': 110,
                                'font-size': "90%",'color': 'grey'
                            },
                        )  
                    ])
                ),

                html.Br(),
                dbc.Row([
                                        
                    dbc.Col([    
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Col([
                                    html.Span(
                                        #id = 'rappel-ruvrement',
                                        style={"color": "red"}),
                                    dbc.Button("VALIDATION", 
                                        #id="button-validation", 
                                        color="primary",block=True),
                                    html.Span(id="example-output",style={'font-size': "90%"}),
                                 ]),
                            ])
                        ),
                    ]),

                ]),
                                
 
             ]          
            return  chld


            
    return ">>>> Sélectioner une rubrique"


# Globale
@app.callback(
    [
        Output("liste-id-sage-credit", "children"),
        Output("liste-id-sage-debit", "children"),
    ],
    [
        Input("dd-dt-ref-sage", "value"),
        Input("dd-dt-var-sage", "value"),
    ]       
)
def card_update(dt,dtv): 
    dt = datetime.datetime.strptime(dt, '%Y-%m-%d').date()
    dtv = datetime.datetime.strptime(dtv, '%Y-%m-%d').date()

    res1 = [      
        xdcc_table(df_byClient_var(dt,dtv)['creditMv']),
    ]

    res2 = [      
        xdcc_table(df_byClient_var(dt,dtv)['debitMv']),
    ]

    return res1, res2


# Debiteur
@app.callback(
    [
        
        Output("sageDebiteur-resume-id", "children"),
        Output("sageDebiteur-table-id", "children"),
        Output("apporteur-graph-id", "children"),
        
    ],
    [
        Input("dd-dt-ref-sage", "value"),
        Input("dd-dt-var-sage", "value"),
        Input("radio-valVar-id","value"),
        Input("dd-apporteur-id","value")
    ]       
)
def update(dt,dtv,rd,app): 
    dt = datetime.datetime.strptime(dt, '%Y-%m-%d').date()
    dtv = datetime.datetime.strptime(dtv, '%Y-%m-%d').date()
    if app == 'TOTAL':
        df_t = df_byClient_var(dt,dtv)['value_bySoldeCur_rec_formated']
    else:
        df_t = recouvrement_filter (df_byClient_var(dt,dtv)['value_bySoldeCur_rec_formated'], 'APPORTEUR', app) 


    if rd == "valeur":
        resume = xdcc_table(debiteur_resume(dt,dtv)['value_formated'])
    else:
        resume = xdcc_table(debiteur_resume(dt,dtv)['variation_formated']) 

    res1 = [resume]

    res2 = [
        xdcc_table(df_t)
    ]

    res3 = solde_graph(app)

    return res1, res2, res3



#Solde par clients
@app.callback(  
       
        Output("SageClient-table-id", "children"),
      
    [
        Input("dd-dt-ref-sage", "value"),
        Input("dd-numClient-id","value"),
    ]
)
def souscripteur_update(dt,c):
    dt = datetime.datetime.strptime(dt, '%Y-%m-%d').date()
    res = [
        html.P('Solde : {:,.0f} MGA '.format(sageClientSolde_table(dt,c)['solde'])),
        xdcc_table(sageClientSolde_table(dt,c)['df'])
    ]

    return res


#debiteur saisie
@app.callback(  
    [   
        Output("input-apporteur-sageRecouvrement-id", "children"),
        Output("input-nom-sageRecouvrement-id", "children"),
        Output("input-solde-sageRecouvrement-id", "children"),
        Output("dd-situation-sageRecouvrement-id", "value"),
        Output("input-commentaire-sageRecouvrement-id", "value"),


    ], 
    [
        Input("dd-numClient-sageRecouvrement-id", "value"),
    ]
)
def update(num):
    if num== None:
        return None, None, None, None, None
    else:
        res = recouvrement_select(num, dt_ref, dt_var)
        res1 = res['apporteur']
        res2 = res['nom']
        res3 = res['solde']
        res4 = res['situation']
        res5 = res['commentaire']

    return res1, res2,res3,res4,res5

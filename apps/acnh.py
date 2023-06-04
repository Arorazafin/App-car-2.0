
# Dash package
import xxlimited
from pandas.io.sql import has_table
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_html_components as html
import dash_table
#from dash.dash_table import FormatTemplate
from dash_table.Format import Format, Scheme, Group
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
import time
#from datetime import datetime


#db package
from sqlalchemy import create_engine
import psycopg2

# local package
from app import app

# les objets
from apps import nhObjet, nhData


today = datetime.date.today()
print("Today's date on acnh page: ", today)


#les agences
nh_agences = {
    100:['ACTIII','100-ACTIII',pd.DataFrame()],
    101:['ACT02','101-ACT02',pd.DataFrame()],
    102:['ACTII','102-ACTII',pd.DataFrame()],
    103:['TOAMASINA','103-TOAMASINA',pd.DataFrame()],
    104:['DCS','104-DCS',pd.DataFrame()],
    105:['INDEPENDANCE','105-INDEPENDANCE',pd.DataFrame()],
    106:['ANTSIRABE','106-ANTSIRABE',pd.DataFrame()],
    107:['FIANARANTSOA','107-FIANARANTSOA',pd.DataFrame()],
    108:['MAHAJANGA','108-MAHAJANGA',pd.DataFrame()],
    109:['SAMBAVA','109-SAMBAVA',pd.DataFrame()],
    110:['ATSIRANANA','110-ATSIRANANA',pd.DataFrame()],
    111:['CAP3000','111-CAP3000',pd.DataFrame()],
    112:['MORONDAVA','112-MORONDAVA',pd.DataFrame()],
    113:['MORAMANGA','113-MORAMANGA',pd.DataFrame()],
    114:['MINORIS','114-MINORIS',pd.DataFrame()],
    115:['TOLIARA','115-TOLIARA',pd.DataFrame()],
    116:['AMBATOLAMPIKELY','116-AMBATOLAMPIKELY',pd.DataFrame()],
    117:['AMBOSITRA','117-AMBOSITRA',pd.DataFrame()],
    118:['ANALAVORY','118-ANALAVORY',pd.DataFrame()],
 #   119:['SIEGE_PRODUCTION','119-SIEGE_PRODUCTION',pd.DataFrame()],
    120:['TOLAGNARO','120-TOLAGNARO',pd.DataFrame()],
    121:['AMBATONDRAZAKA','121-AMBATONDRAZAKA',pd.DataFrame()],
    122:['NOSYBE','122-NOSYBE',pd.DataFrame()],
    123:['ANTALAHA','123-ANTALAHA',pd.DataFrame()],
    124:['ITAOSY','124-ITAOSY',pd.DataFrame()],
    125:['ANTSOHIHY','125-ANTSOHIHY',pd.DataFrame()],
    126:['MANAKARA','126-MANAKARA',pd.DataFrame()],
    200:['RAR','200-RAR',pd.DataFrame()],
     201:['CAR','201-CAR',pd.DataFrame()],
    207:['SAMASS','207-SAMASS',pd.DataFrame()],
    209:['AFINE','209-AFINE',pd.DataFrame()],
    212:['AVENIR','212-AVENIR',pd.DataFrame()],
    
}

#les constantes
lsIndicateur = ['prime','solde','tx_paiement','ranking']
lsAgencesDropDown = {0:['Toutes Agences','Toutes Agences',None]}
lsAgencesDropDown.update(nh_agences)



#get env db
host=os.environ.get('HOST')
user=os.environ.get('USER')
database=os.environ.get('DATABASE')
pwd=os.environ.get('PASSWORD')


#constante :


db_url = "postgresql://nwyogohsbuhbmx:cb4c6f636ed404b05544a72f064a31b146fada7737552200ab96e5eb9eaa37ff@ec2-34-247-118-233.eu-west-1.compute.amazonaws.com:5432/d9ssmvh4i7116e"
#db_url = 'postgresql://'+user+':'+pwd+'@'+host+'/'+database 
#db_url = os.environ.get('URL')
print(db_url)

# Create an engine instance
alchemyEngine = create_engine(db_url)

# Connect to PostgreSQL server
dbConnection = alchemyEngine.connect()
# Read data from PostgreSQL database table and load into a DataFrame instance

for k,v in nh_agences.items():
    try:
        requete = 'SELECT * FROM "'+v[0]+'"' 
        v[2]  = pd.read_sql(requete, dbConnection)
    except:
        v[2]=pd.DataFrame()

#Close the database connection
dbConnection.close()

print("welcome acnh page")

#for k,v in nh_agences.items():
#        print(k)
#        print(v[2])
#        print(" ")



import sys
print("mem acnh page: ", sys.getsizeof(nh_agences))
print()



# nettoyage de la base 

checking =  nhData.func_checking(nh_agences)
print('data erreur: '+ str(checking ['erreur']))

nh_agences = nhData.func_reguler(checking['df'],nh_agences)

checking =  nhData.func_checking(nh_agences)
print('data erreur après correction: '+ str(checking ['erreur']))

nh_agences = nhData.change_type(nh_agences)
print('changement types: OK')




# Les objets et les collections

## appel des instances  

t_start = time.perf_counter()
print("Début chargement .... ")
#print(t_start)
instAgences = nhObjet.Agences(nh_agences)
instAgences.initCollection()
t_end = time.perf_counter()
deltaT= t_end - t_start
minutes, secondes = divmod(deltaT, 60)
print('-Table created on {:02d}mn{:02d}sec'.format(int(minutes),int(secondes)))

#les elements du reseau
dateExtractionReseau = instAgences.collectionAgences[next(iter(instAgences.collectionAgences))].dateExtraction
print()
print("nb agences: "+ str(len(instAgences.collectionAgences)))
print("Date d'extraction: ")
print(dateExtractionReseau)
print()



def func_instAgence(codeAgence):

    instAgence = instAgences.collectionAgences[codeAgence]
    
    
    dateExtraction =  instAgence.dateExtraction  
    agenceCode = instAgence.CodeNomAgence
  
    instQuittances = instAgence.quittances

    instAssurances = instAgence.assurances

    instClients =  instAgence.clients
    verificationCoherence = instAgence.verificationCoherence

    res = dict()
    res = {
        "dateExtraction":dateExtraction,
        "agenceCode":agenceCode,
        "instQuittances":instQuittances,
        "instAssurances":instAssurances,
        "instClients":instClients,
        "verificationCoherence":verificationCoherence
    }

    return res



# les fonctions pour la viz

## vue quittance
## definition des variables à utiliser


def card_content(var,chiffre, format):
    
    if format == "%" :
        res = [
            dbc.CardHeader(var),
            dbc.CardBody(
                [
                    html.H5('{:.0%}'.format(chiffre), className="card-title"),
                ]
            ),
        ]

    else:    

        res = [
            dbc.CardHeader(var),
            dbc.CardBody(
                [
                    html.H5('{:,.0f}'.format(chiffre), className="card-title"),
                ]
            ),
        ]
    return res

def card_content2(titre,var,nb,nb_p,prime,prime_p,tx):
    

    res = [
        dbc.CardHeader(titre),
        dbc.CardBody(
            [
                html.H5(var, className="card-title"),
                html.P("Nombre: "+str('{:,.0f}'.format(nb))+" ( " + str('{:.0%}'.format(nb_p))+" du total)", className="card-text"),
                html.P("Prime: "+str('{:,.0f}'.format(prime))+ "( "+ str('{:.0%}'.format(prime_p))+" du total)"),
                html.P("Taux de paiement: "+str('{:.0%}'.format(tx))),

            ]
        ),
    ]

    return res



def graph_evolQuittance_prime(instQ):

    df_graph = instQ.evolAnnuel

    df_graph = df_graph[df_graph.index>2019]
    df_graph = df_graph[df_graph.index<2023]
    
    fig = go.Figure()

    fig.add_trace(
    go.Bar(x=df_graph.index,
           y=df_graph['prime'],
           name = "Prime d'assurances"
        )
    )


    fig.add_trace(
        go.Bar(x=df_graph.index,
            y=df_graph['paiement'],
            name = "Paiement effectif"
        )
    )
    fig.update_layout(title_text="Dynamique Prime & Paiement")
    fig.update_xaxes(
            showgrid=True,
            ticks="outside",
            tickson="boundaries",
            ticklen=20,
            ticktext=["2020", "2021", "2022 YTD"],
            tickvals=[2020, 2021, 2022],
        )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    dcc_graph = dcc.Graph(figure = fig)


    return dcc_graph

def graph_evolQuittance_solde_agences(instAgences,reseau):

    df_graph = instAgences.vueQuittancesEvolAnnuel[reseau]
    df_graph = df_graph[df_graph.index>2019]
    df_graph = df_graph[df_graph.index<2023]
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df_graph.index,
            y=df_graph['solde'],
            name = "Prime à recouvrer"
        )
    )
    fig.update_layout(title_text="Prime à recouvrer")
    fig.update_xaxes(
            showgrid=True,
            ticks="outside",
            tickson="boundaries",
            ticklen=20,
            ticktext=["2020", "2021", "2022 YTD"],
            tickvals=[2020, 2021, 2022],
        )
    
    dcc_graph = dcc.Graph(figure = fig)
    
    return dcc_graph

def graph_evolQuittance_solde(instQ):

    df_graph = instQ.evolAnnuel
    df_graph = df_graph[df_graph.index>2019]
    df_graph = df_graph[df_graph.index<2023]
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df_graph.index,
            y=df_graph['solde'],
            name = "Prime à recouvrer"
        )
    )
    fig.update_layout(title_text="Prime à recouvrer")
    fig.update_xaxes(
            showgrid=True,
            ticks="outside",
            tickson="boundaries",
            ticklen=20,
            ticktext=["2020", "2021", "2022 YTD"],
            tickvals=[2020, 2021, 2022],
        )
    
    dcc_graph = dcc.Graph(figure = fig)
    
    return dcc_graph


def graph_EvolQuittance_prime_agences(instAgences,reseau):

    df_graph = instAgences.vueQuittancesEvolAnnuel[reseau]
    
    df_graph = df_graph[df_graph.index>2019]
    df_graph = df_graph[df_graph.index<2023]
    
    fig = go.Figure()

    fig.add_trace(
    go.Bar(x=df_graph.index,
           y=df_graph['prime'],
           name = "Prime d'assurances"
        )
    )


    fig.add_trace(
        go.Bar(x=df_graph.index,
            y=df_graph['paiement'],
            name = "Paiement effectif"
        )
    )
    fig.update_layout(title_text="Dynamique Prime & Paiement")
    fig.update_xaxes(
            showgrid=True,
            ticks="outside",
            tickson="boundaries",
            ticklen=20,
            ticktext=["2020", "2021", "2022 YTD"],
            tickvals=[2020, 2021, 2022],
        )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    dcc_graph = dcc.Graph(figure = fig)


    return dcc_graph
## vue Assurance


def card_content2(titre,var,nb,nb_p,prime,prime_p,tx):
    

    res = [
        dbc.CardHeader(titre),
        dbc.CardBody(
            [
                html.H5(var, className="card-title"),
                html.P("Nombre: "+str('{:,.0f}'.format(nb))+" ( " + str('{:.0%}'.format(nb_p))+" du total)", className="card-text"),
                html.P("Prime: "+str('{:,.0f}'.format(prime))+ "( "+ str('{:.0%}'.format(prime_p))+" du total)"),
                html.P("Taux de paiement: "+str('{:.0%}'.format(tx))),

            ]
        ),
    ]

    return res


def graph_assurance (instAssurances,assurance_type,assurance_y):

    if assurance_type == 'categorie':
        df_graphAssurance = instAssurances.dfCategorie
    else:
        df_graphAssurance = instAssurances.dfBranche

    if assurance_y == 'solde':
        df_graphAssurance = df_graphAssurance[df_graphAssurance['solde']<=0].copy()
        df_graphAssurance.sort_values(by=assurance_y, ascending = True, inplace = True)
    elif assurance_y == 'tx_paiement':
        df_graphAssurance = df_graphAssurance[df_graphAssurance['solde']<=0].copy()
        df_graphAssurance.sort_values(by=assurance_y, ascending = False, inplace = True)
        
    
    else:
        df_graphAssurance.sort_values(by=assurance_y, ascending = False, inplace = True)


    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df_graphAssurance.index,
            y=df_graphAssurance[assurance_y],
        )
    )
    fig.update_layout(
        title_text= "Répartition de '" + assurance_y+ "' sur chaque type de '" + assurance_type + "'"             
    )

    dcc_graph = dcc.Graph(figure = fig)
    return dcc_graph

def graph_assurance_agences (instAgenceAssurances,assurance_type,assurance_y,reseau):

    if assurance_type == 'categorie':
        df_graphAssurance = instAgenceAssurances.vueAssurancesDfCategorie[reseau]
    else:
        df_graphAssurance = instAgenceAssurances.vueAssurancesDfBranche[reseau]

    if assurance_y == 'solde':
        df_graphAssurance = df_graphAssurance[df_graphAssurance['solde']<=0].copy()
        df_graphAssurance.sort_values(by=assurance_y, ascending = True, inplace = True)
    elif assurance_y == 'tx_paiement':
        df_graphAssurance = df_graphAssurance[df_graphAssurance['solde']<=0].copy()
        df_graphAssurance.sort_values(by=assurance_y, ascending = False, inplace = True)
        
    
    else:
        df_graphAssurance.sort_values(by=assurance_y, ascending = False, inplace = True)


    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df_graphAssurance.index,
            y=df_graphAssurance[assurance_y],
        )
    )
    fig.update_layout(
        title_text= "Répartition de '" + assurance_y+ "' sur chaque type de '" + assurance_type + "'"             
    )

    dcc_graph = dcc.Graph(figure = fig)
    return dcc_graph


## Vue client
def card_content3(titre,nom,prime,prime_p,tx, solde):
    

    res = [
        dbc.CardHeader(titre),
        dbc.CardBody(
            [
                html.H5(nom, className="card-title"),
                html.P("Prime: "+str('{:,.0f}'.format(prime))+ "( "+ str('{:.0%}'.format(prime_p))+" du total)"),
                html.P("Taux de paiement: "+str('{:.0%}'.format(tx))),
                html.P("Solde: "+str('{:,.0f}'.format(solde))),

            ]
        ),
    ]

    return res

def graph_client (instC,client_y):

    #client_y = 'solde'

    df_graphclient = instC.dfClient

    if client_y == 'solde':
        df_graphclient = df_graphclient[df_graphclient['solde']<=0].copy()
        df_graphclient.sort_values(by=client_y, ascending = True, inplace = True)
    elif client_y == 'tx_paiement':
        df_graphclient = df_graphclient[df_graphclient['solde']<=0].copy()
        df_graphclient.sort_values(by=client_y, ascending = False, inplace = True)
    else:
        df_graphclient.sort_values(by=client_y, ascending = False, inplace = True)

    df_graphclient = df_graphclient.head(20)  

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df_graphclient['nom'],
            y=df_graphclient[client_y],
        )
    )
    fig.update_layout(
        title_text= "Répartition de '" + client_y+ "' par clients"         ,             
    )    


    dcc_graph = dcc.Graph(figure = fig)
    return dcc_graph

def graph_client_agence (instAgenceClient,client_y,reseau):

    #client_y = 'solde'

    df_graphclient = instAgenceClient.vueClientsDfClient[reseau]

    if client_y == 'solde':
        df_graphclient = df_graphclient[df_graphclient['solde']<=0].copy()
        df_graphclient.sort_values(by=client_y, ascending = True, inplace = True)
    elif client_y == 'tx_paiement':
        df_graphclient = df_graphclient[df_graphclient['solde']<=0].copy()
        df_graphclient.sort_values(by=client_y, ascending = False, inplace = True)
    else:
        df_graphclient.sort_values(by=client_y, ascending = False, inplace = True)

    df_graphclient = df_graphclient.head(20)  

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df_graphclient['nom'],
            y=df_graphclient[client_y],
        )
    )
    fig.update_layout(
        title_text= "Répartition de '" + client_y+ "' par clients"         ,             
    )    


    dcc_graph = dcc.Graph(figure = fig)
    return dcc_graph





def func_table (dfx):
    
  
    percentage = Format(precision=0, scheme=Scheme.percentage)
    formatTemp = Format(
            scheme=Scheme.fixed, 
            precision=0,
            group=Group.yes,
            groups=3,
        )


    dcc_table = dash_table.DataTable(
        #id='table',
        #columns=[{"name": i, "id": i} for i in dfx.columns],
    
        
        columns = [
            dict(id='codeName', name='Agence'),
            dict(id='prime', name='Prime', type='numeric', format=formatTemp),
            dict(id='solde', name='Solde', type='numeric', format=formatTemp),
            dict(id='tx_paiement', name='Tx_paiement', type='numeric', format=percentage),
            dict(id='ranking', name='Ranking', type='numeric', format=formatTemp),
        ],
        
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
        ],
        sort_action='native',


    )
    return dcc_table


layout = dbc.Container(
    [
        #dcc.Store(id="store"),
        #html.H1("Cabinet d'Assurance Razafindrakola - Portefeuille"),
        #html.Hr(),
        dbc.Row([
            
            dbc.Col([                  
   
                #html.Br(),
                dbc.Col([
                    html.Br(),
                    dcc.Dropdown(
                            id='idDropdownAgences',
                            options=[
                                {'label': v[1], 'value': k} for k,v in lsAgencesDropDown.items()
                            ],
                            #optionHeight=45,
                            value = 0,
                            #placeholder="Selectionner une agence",
                            clearable = False,
                            # style={
                            #     'font-size': "100%",
                            #    ' background-color': '#F08080',
                            # },
                            
                        ),

                    html.Br(), 
                    dbc.Card([
                        dbc.CardBody(
                            [
                                html.P("Agence",style={'text-align': 'center','font-weight': 'bold'},className="card-subtitle"),
                                html.P(style={'text-align': 'center'},className="card-text", 
                                    id="idCardNomAgence"),
                                                                
                            ]
                        ),

                    ]),

                    
                    html.Br(),
                    dbc.Card([
                        dbc.CardBody(
                            [
                                html.P("Date d'extraction",style={'text-align': 'center','font-weight': 'bold'},className="card-subtitle"),
                                html.P(style={'text-align': 'center'},className="card-text",
                                      id="idDateExtraction"  
                                ),
                                html.P('Historique: ',style={ 'font-size': '10px', 'font-weight': 'bold'},className="card-subtitle"),
                                html.P('depuis 01-01-2020',style={ 'font-size': '10px'},className="card-subtitle"),
                                html.P('-',style={ 'font-size': '10px'},className="card-subtitle"),
                                html.P('Vérification des données: ',style={ 'font-size': '10px', 'font-weight': 'bold'},className="card-subtitle"),
                                html.P(style={ 'font-size': '10px'},className="card-text",
                                      id="idVerif"  
                                ),
                                
                            ]
                        ),

                    ]),
                    

                    html.Br(),
                    dbc.Card([
                        dbc.CardBody(
                            [
                                html.P('Indicateurs',className="card-subtitle",style={'text-align': 'center','font-weight': 'bold'}),
                                html.P(),
                                html.P('Prime: ',className="card-subtitle",style={'font-weight': 'bold', 'font-size': '10px'}),
                                html.P("Prime d'assurances associée au contrat",style={ 'font-size': '10px'},className="card-text"),
                                html.P('Solde: ',className="card-subtitle",style={'font-weight': 'bold', 'font-size': '10px'}),
                                html.P("Encaissement-Prime. Un chiffre négatif représente un impayé",style={ 'font-size': '10px'},className="card-text"),
                                html.P('Taux de paiement: ',className="card-subtitle",style={'font-weight': 'bold', 'font-size': '10px'}),
                                html.P("Encaissement/Prime. 100%=prime entièrement payée",style={ 'font-size': '10px'},className="card-text"),
                                html.P('Ranking: ',className="card-subtitle",style={'font-weight': 'bold', 'font-size': '10px'}),
                                html.P("prime*tauxPaiement. Indicateur de rentabilité et de performance",style={ 'font-size': '10px'},className="card-text"),
                               
                            ]
                        ),

                    ]),

                    html.Br(),
                    dbc.Card([
                        dbc.CardBody(
                            [
                                html.P('Définition',className="card-subtitle", style={'text-align': 'center','font-weight': 'bold'}),
                                html.P(),
                                html.P('YTD: ', style={'font-weight': 'bold', 'font-size': '10px'},className="card-subtitle"), 
                                html.P("Year-To-Date. Cumul des données du debut de l'année jusqu'à la date d'extraction", 
                                    style={ 'font-size': '10px'},
                                    className="card-text"),       
                            ]
                        ),

                    ]),
                    
                    
                ]),
                
                html.Br(),
            

            ],xs=12, sm=6, md=6, lg=2, xl=2),

            dbc.Col([
                dbc.Tabs(
                [
                    dbc.Tab(label="Page d'acceuil", tab_id="idHome3"),
                    dbc.Tab(label="Vue Quittance", tab_id="idQuittance3"),
                    dbc.Tab(label="Vue Assurance", tab_id="idAssurance3"),
                    dbc.Tab(label="Vue Client", tab_id="idClient3"),
                   

                ],
                id="id-tabs-v23",
                active_tab="idHome3",
                ),
                html.Br(),
                html.Div(id="id-tab-content-v23"),#, className="p-4"),
                html.Div(id="idRadioReseau")
            ],xs=12, sm=6, md=6, lg=10, xl=10
            ),

            
        ]),
    ],fluid=True
)

#callback tab
@app.callback(
    [
        Output("id-tab-content-v23", "children"),
        Output("idCardNomAgence","children"),
        Output("idDateExtraction","children"),
        Output("idVerif","children"),

    ],
    [
         Input("id-tabs-v23", "active_tab"),
         Input("idDropdownAgences","value"),
         Input("idRadioReseau","value"),

    ]
    
    
)
def render_tab_content(active_tab,v1,v2):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """

    if v1==0:
        instAgence = instAgences
    else:
        instAgence = func_instAgence(v1)
 
    
    if active_tab is not None:
        if active_tab == "idHome3":

            if v1==0:
                chld = [

                    dbc.Row(
                        #dbc.Col("Analyse sur un Réseau de {} agences, dont {} Agences Centrales et {} Agences Générales".format(instAgences.totalNombreReseau['all'],instAgences.totalNombreReseau['ac'],instAgences.totalNombreReseau['ag']),
                        #id = 'idRadioReseau',
                        #style={ 'font-size': '10px', 'font-weight': 'bold'}
                        #)
                        dbc.Col([
                            html.Span('Analyse sur un Réseau de '),
                            html.Strong(instAgences.totalNombreReseau['all']),
                            html.Span(' agences, dont '),
                            html.Strong(instAgences.totalNombreReseau['ac']),
                            html.Span(' Agences Centrales et '),
                            html.Strong(instAgences.totalNombreReseau['ag']),
                            html.Span(' Agences Générales.'),
                            ],id = 'idRadioReseau',)
                    ),
                    html.Hr(),


                    dbc.Row(dbc.Col('Toutes agences confondues', className='h4')),
                    dbc.Row(dbc.Col(
                        func_table(instAgence.tableauSynthese['all']),
                        xs=12, sm=12, md=12, lg=12, xl=12)
                    ),
                    html.Br(),
                    dbc.Row(dbc.Col('Agences Centrales', className='h4')),
                    dbc.Row(dbc.Col(
                        func_table(instAgence.tableauSynthese['ac']),
                        xs=12, sm=12, md=12, lg=12, xl=12)
                    ),
                    html.Br(),
                    dbc.Row(dbc.Col('Agences Générales', className='h4')),
                    dbc.Row(dbc.Col(
                        func_table(instAgence.tableauSynthese['ag']),
                        xs=12, sm=12, md=12, lg=12, xl=12)
                    ),
                                       
                   

                ]
                res00 = "Toutes Agences"
                res01 = dateExtractionReseau
                res02 = None
            else:  
                chld = [
                    dbc.Row(dbc.Col(id = 'idRadioReseau')),
                    html.Hr(),
                    html.P('>>>>> Selectionner une rubrique')
                    ]

                res00 = instAgence['agenceCode']
                res01 = instAgence['dateExtraction']
                res02 = instAgence['verificationCoherence']

            return  chld, res00,res01, res02
        
        elif active_tab == "idQuittance3":
            #quittances

            if v1 == 0:
                instAgencesQuittances = instAgences

                if v2 == None:
                    reseau = 'all'
                else:
                    reseau = v2

                res00 = "Toutes Agences"
                res01 = dateExtractionReseau
                res02 = None

                chld = [

                    dbc.Row([
                        dbc.Col([
                            dcc.RadioItems(
                                id = 'idRadioReseau',
                                options=[ 
                                        {'label': i, 'value': j} for i,j in zip(['Toutes Agences', 'Agence Centrale', 'Agence Générale'],['all', 'ac','ag'])
                                ],
                                value=reseau,
                                labelStyle={'display': 'inline-block'},
                                inputStyle={"margin-left": "10px"},
                                
                            ),
                        ], xs=12, sm=12, md=12, lg=12, xl=12)
                    ]),
                    
                    html.Hr(),

                    dbc.Row(dbc.Col('Chiffres clés des Quittances émises', className='h5')),
                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card(card_content("Primes",instAgencesQuittances.totalPrime[reseau],"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id='cardQ11')
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Solde",instAgencesQuittances.totalSolde[reseau],"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id='cardQ12')
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Nombre de quittances",instAgencesQuittances.vueQuittancesNombre[reseau],"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id='cardQ13')
                            ],xs=12, sm=12, md=12, lg=4, xl=4),
                        ],
                    ),
                    html.Br(),
                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card(card_content("Primes 2021",instAgencesQuittances.vueQuittancesEvolAnnuel2(reseau,2021,'prime'),"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ21")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Solde 2021",instAgencesQuittances.vueQuittancesEvolAnnuel2(reseau,2021,'solde'),"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ22")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Taux de paiement 2021",instAgencesQuittances.vueQuittancesEvolAnnuel2(reseau,2021,'tx_paiement'),"%"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ23")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),
                        ],

                    ),
                    html.Br(),
                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card(card_content("Primes 2022 (YTD)",instAgencesQuittances.vueQuittancesEvolAnnuel2(reseau,2022,'prime'),"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ31")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Solde 2022 (YTD)",instAgencesQuittances.vueQuittancesEvolAnnuel2(reseau,2022,'solde'),"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ32")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Taux de paiement 2022 (YTD)",instAgencesQuittances.vueQuittancesEvolAnnuel2(reseau,2022,'tx_paiement'),"%"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ33")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),
                        ],

                    ),
                    html.Hr(),
                    dbc.Row(dbc.Col("Représentation graphique d'évolution des réalisations", className='h5')),
                    dbc.Row(
                        [
                            dbc.Col([graph_EvolQuittance_prime_agences(instAgencesQuittances,reseau)],id ="idGraphQuitEvolPrime", xs=12, sm=12, md=12, lg=6, xl=6),
                            dbc.Col([graph_evolQuittance_solde_agences(instAgencesQuittances,reseau)],id ="idGraphQuitEvolSolde", xs=12, sm=12, md=12, lg=6, xl=6),

                        ],
                    )
                ]
            else:

                instQuittances = instAgence["instQuittances"]
                res00 = instAgence['agenceCode']
                res01 = instAgence['dateExtraction']
                res02 = instAgence['verificationCoherence']
                
                chld = [
                    dbc.Row(dbc.Col(id = 'idRadioReseau')),
                    html.Hr(),
                    dbc.Row(dbc.Col('Chiffres clés des Quittances émises', className='h5')),
                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card(card_content("Primes",instQuittances.totalPrime,"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id='cardQ11')
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Solde",instQuittances.totalSolde,"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id='cardQ12')
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Nombre de quittances",instQuittances.totalNombre,"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id='cardQ13')
                            ],xs=12, sm=12, md=12, lg=4, xl=4),
                        ],
                    ),
                    html.Br(),
                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card(card_content("Primes 2021",instQuittances.evolAnnuel2(2021,'prime'),"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ21")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Solde 2021",instQuittances.evolAnnuel2(2021,'solde'),"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ22")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Taux de paiement 2021",instQuittances.evolAnnuel2(2021,'tx_paiement'),"%"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ23")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),
                        ],

                    ),
                    html.Br(),
                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card(card_content("Primes 2022 (YTD)",instQuittances.evolAnnuel2(2022,'prime'),"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ31")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Solde 2022 (YTD)",instQuittances.evolAnnuel2(2022,'solde'),"nb"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ32")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(card_content("Taux de paiement 2022 (YTD)",instQuittances.evolAnnuel2(2022,'tx_paiement'),"%"),
                                style={'text-align': 'center'}, color="danger", outline=True,
                                id="cardQ33")
                            ],xs=12, sm=12, md=12, lg=4, xl=4),
                        ],

                    ),
                    html.Hr(),
                    dbc.Row(dbc.Col("Représentation graphique d'évolution des réalisations", className='h5')),
                    dbc.Row(
                        [
                            dbc.Col([graph_evolQuittance_prime(instQuittances)],id ="idGraphQuitEvolPrime", xs=12, sm=12, md=12, lg=6, xl=6),
                            dbc.Col([graph_evolQuittance_solde(instQuittances)],id ="idGraphQuitEvolSolde", xs=12, sm=12, md=12, lg=6, xl=6),

                        ],
                    ),


                ]
            
            
            return  chld, res00,res01, res02


        elif active_tab == "idAssurance3":
            if v1 == 0:
                instAgencesAssurances = instAgences

                if v2 == None:
                    reseau = 'all'
                else:
                    reseau = v2

                res00 = "Toutes Agences"
                res01 = dateExtractionReseau
                res02 = None

                chld = [

                    dbc.Row([
                        dbc.Col([
                            dcc.RadioItems(
                                id = 'idRadioReseau',
                                options=[ 
                                        {'label': i, 'value': j} for i,j in zip(['Toutes Agences', 'Agence Centrale', 'Agence Générale'],['all', 'ac','ag'])
                                ],
                                value=reseau,
                                labelStyle={'display': 'inline-block'},
                                inputStyle={"margin-left": "10px"},
                                
                            ),
                        ], xs=12, sm=12, md=12, lg=12, xl=12)
                    ]),
                    html.Hr(),

                    dbc.Row(dbc.Col("Chiffres clés des Contrats d'Assurances", className='h5')),
                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card(
                                (card_content("Primes",instAgencesAssurances.vueAssurancesTotalPrime[reseau],"nb")),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(
                                (card_content("Solde",instAgencesAssurances.vueAssurancesTotalSolde[reseau],"nb")),style={'text-align': 'center'}, color="danger", outline=True)
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(
                                (card_content("Nombre de contrats",instAgencesAssurances.vueAssurancesTotalNombre[reseau],"nb")),style={'text-align': 'center'}, color="danger", outline=True)
                            ],xs=12, sm=12, md=12, lg=4, xl=4),
                        ],
                    ),
                    html.Br(),

                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card((
                                card_content2("Meilleur Produit - Branche",
                                    instAgencesAssurances.vueAssurancesBrancheRentable[reseau]['product'],
                                    instAgencesAssurances.vueAssurancesBrancheRentable[reseau]['nb'],
                                    instAgencesAssurances.vueAssurancesBrancheRentable[reseau]['nbPercentage'],
                                    instAgencesAssurances.vueAssurancesBrancheRentable[reseau]['prime'],
                                    instAgencesAssurances.vueAssurancesBrancheRentable[reseau]['primePercentage'],
                                    instAgencesAssurances.vueAssurancesBrancheRentable[reseau]['txPaiement'],
                                )
                                ),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=6, xl=6),

                            dbc.Col([dbc.Card((
                                card_content2("Meilleur Produit - Catégorie",
                                    instAgencesAssurances.vueAssurancesCategorieRentable[reseau]['product'],
                                    instAgencesAssurances.vueAssurancesCategorieRentable[reseau]['nb'],
                                    instAgencesAssurances.vueAssurancesCategorieRentable[reseau]['nbPercentage'],
                                    instAgencesAssurances.vueAssurancesCategorieRentable[reseau]['prime'],
                                    instAgencesAssurances.vueAssurancesCategorieRentable[reseau]['primePercentage'],
                                    instAgencesAssurances.vueAssurancesCategorieRentable[reseau]['txPaiement'],
                                )
                                ),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=6, xl=6),

                        
                        ],
                    ),
                    html.Hr(),
                    dbc.Row(dbc.Col("Représentation graphique des Contrats d'Assurances", className='h5')),
                    dbc.Row([
                        dbc.Col([
                            dcc.RadioItems(
                                id = 'idRadioAssurance3',
                                options=[
                                        {'label': i, 'value': j} for i,j in zip(['Branche', 'Catégorie'],['branche', 'categorie'])
                                ],
                                value='branche',
                                labelStyle={'display': 'inline-block'},
                                inputStyle={"margin-left": "10px"},
                                
                            ),
                        ], xs=12, sm=12, md=12, lg=4, xl=4),
                            
                    ]),
                    dbc.Row([    
                        dbc.Col([
                            html.P(">>>> Sélection de l'indicateur: "),
                            dcc.Dropdown(
                                id='idDropdownAssurance3',
                                options=[
                                    {'label': i, 'value': i} for i in lsIndicateur
                                ],
                                optionHeight=45,
                                value = lsIndicateur[0],
                                clearable = False,
                                style={
                                    'font-size': "90%",
                                },
                            )
                        ], xs=12, sm=12, md=12, lg=4, xl=4)
                    ]),

                
                    dbc.Row(
                        dbc.Col(
                            ##graph_assurance (instAssurances,v1,v2),
                            id = 'idGraphAssurance3',
                            xs=12, sm=12, md=12, lg=12, xl=12  
                        ),
                    ),
                    html.Br(),
                    
                            
                


                ]
            else:
            
                instAssurances = instAgence["instAssurances"]
                res00 = instAgence['agenceCode']
                res01 = instAgence['dateExtraction']
                res02 = instAgence['verificationCoherence']

                chld =  [
                    dbc.Row(dbc.Col(id = 'idRadioReseau')),
                    html.Hr(),
                    
                    dbc.Row(dbc.Col("Chiffres clés des Contrats d'Assurances", className='h5')),
                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card(
                                (card_content("Primes",instAssurances.totalPrime,"nb")),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(
                                (card_content("Solde",instAssurances.totalSolde,"nb")),style={'text-align': 'center'}, color="danger", outline=True)
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(
                                (card_content("Nombre de contrats",instAssurances.totalNombre,"nb")),style={'text-align': 'center'}, color="danger", outline=True)
                            ],xs=12, sm=12, md=12, lg=4, xl=4),
                        ],
                    ),
                    html.Br(),

                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card((
                                card_content2("Meilleur Produit - Branche",
                                    instAssurances.brancheRentable['product'],
                                    instAssurances.brancheRentable['nb'],
                                    instAssurances.brancheRentable['nbPercentage'],
                                    instAssurances.brancheRentable['prime'],
                                    instAssurances.brancheRentable['primePercentage'],
                                    instAssurances.brancheRentable['txPaiement'],
                                )
                                ),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=6, xl=6),

                            dbc.Col([dbc.Card((
                                card_content2("Meilleur Produit - Catégorie",
                                    instAssurances.categorieRentable['product'],
                                    instAssurances.categorieRentable['nb'],
                                    instAssurances.categorieRentable['nbPercentage'],
                                    instAssurances.categorieRentable['prime'],
                                    instAssurances.categorieRentable['primePercentage'],
                                    instAssurances.categorieRentable['txPaiement'],
                                )
                                ),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=6, xl=6),

                        
                        ],
                    ),
                    html.Hr(),
                    dbc.Row(dbc.Col("Représentation graphique des Contrats d'Assurances", className='h5')),
                    dbc.Row([
                        dbc.Col([
                            dcc.RadioItems(
                                id = 'idRadioAssurance3',
                                options=[
                                        {'label': i, 'value': j} for i,j in zip(['Branche', 'Catégorie'],['branche', 'categorie'])
                                ],
                                value='branche',
                                labelStyle={'display': 'inline-block'},
                                inputStyle={"margin-left": "10px"},
                                
                            ),
                        ], xs=12, sm=12, md=12, lg=4, xl=4),
                            
                    ]),
                    dbc.Row([    
                        dbc.Col([
                            html.P(">>>> Sélection de l'indicateur: "),
                            dcc.Dropdown(
                                id='idDropdownAssurance3',
                                options=[
                                    {'label': i, 'value': i} for i in lsIndicateur
                                ],
                                optionHeight=45,
                                value = lsIndicateur[0],
                                clearable = False,
                                style={
                                    'font-size': "90%",
                                },
                            )
                        ], xs=12, sm=12, md=12, lg=4, xl=4)
                    ]),

                
                    dbc.Row(
                        dbc.Col(
                            ##graph_assurance (instAssurances,v1,v2),
                            id = 'idGraphAssurance3',
                            xs=12, sm=12, md=12, lg=12, xl=12  
                        ),
                    ),
                    html.Br(),
                    
                            
                
                ]     

                
            return  chld,res00,res01, res02

        elif active_tab == "idClient3":

            if v1 == 0:
                instAgencesClients = instAgences

                if v2 == None:
                    reseau = 'all'
                else:
                    reseau = v2

                res00 = "Toutes Agences"
                res01 = dateExtractionReseau
                res02 = None

                chld = [

                    dbc.Row([
                        dbc.Col([
                            dcc.RadioItems(
                                id = 'idRadioReseau',
                                options=[ 
                                        {'label': i, 'value': j} for i,j in zip(['Toutes Agences', 'Agence Centrale', 'Agence Générale'],['all', 'ac','ag'])
                                ],
                                value=reseau,
                                labelStyle={'display': 'inline-block'},
                                inputStyle={"margin-left": "10px"},
                                
                            ),
                        ], xs=12, sm=12, md=12, lg=12, xl=12)
                    ]),
                    html.Hr(),

                    dbc.Row(dbc.Col('Chiffres clés des Souscripteurs', className='h5')),
                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card(
                                (card_content("Primes",instAgencesClients.vueClientsTotalPrime[reseau],"nb")),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(
                                (card_content("Solde",instAgencesClients.vueClientsTotalSolde[reseau],"nb")),style={'text-align': 'center'}, color="danger", outline=True)
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(
                                (card_content("Nombre de clients",instAgencesClients.vueClientsTotalNombre[reseau],"nb")),style={'text-align': 'center'}, color="danger", outline=True)
                            ],xs=12, sm=12, md=12, lg=4, xl=4),
                        ],
                    ),
                    html.Br(),

                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card((
                                card_content3("Meilleur client",
                                    instAgencesClients.vueClientsClientPlusRentable[reseau]['nom'],
                                    instAgencesClients.vueClientsClientPlusRentable[reseau]['prime'],
                                    instAgencesClients.vueClientsClientPlusRentable[reseau]['primePercentage'],
                                    instAgencesClients.vueClientsClientPlusRentable[reseau]['txPaiement'],
                                    instAgencesClients.vueClientsClientPlusRentable[reseau]['solde'],

                                )
                                ),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card((
                                card_content3("2e Meilleur client",
                                    instAgencesClients.vueClientsClient2ePlusRentable[reseau]['nom'],
                                    instAgencesClients.vueClientsClient2ePlusRentable[reseau]['prime'],
                                    instAgencesClients.vueClientsClient2ePlusRentable[reseau]['primePercentage'],
                                    instAgencesClients.vueClientsClient2ePlusRentable[reseau]['txPaiement'],
                                    instAgencesClients.vueClientsClient2ePlusRentable[reseau]['solde'],

                                )
                                ),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card((
                                card_content3("Recouvrement client",
                                    instAgencesClients.vueClientsClientPlusGrosDeficit[reseau]['nom'],
                                    instAgencesClients.vueClientsClientPlusGrosDeficit[reseau]['prime'],
                                    instAgencesClients.vueClientsClientPlusGrosDeficit[reseau]['primePercentage'],
                                    instAgencesClients.vueClientsClientPlusGrosDeficit[reseau]['txPaiement'],
                                    instAgencesClients.vueClientsClientPlusGrosDeficit[reseau]['solde'],

                                )
                                ),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=4, xl=4),



                        
                        ],
                    ),
                    html.Hr(),
                    dbc.Row(dbc.Col("Représentation graphique des Souscripteurs", className='h5')),
                    dbc.Row([
                        dbc.Col([
                            html.P(">>>> Sélection de l'indicateur: "),
                            dcc.Dropdown(
                                id='idDropdownClient3',
                                options=[
                                    {'label': i, 'value': i} for i in lsIndicateur
                                ],
                                optionHeight=45,
                                value = lsIndicateur[0],
                                clearable = False,
                                style={
                                    'font-size': "90%",
                                },
                            )
                        ], xs=12, sm=12, md=12, lg=4, xl=4)
                    ]),

                
                    dbc.Row(
                        dbc.Col(
                            id = 'idGraphClient3',
                            xs=12, sm=12, md=12, lg=12, xl=12  
                        ),
                    ),
                    html.Br(),
                    
                



                ]
            else:  

                instClients = instAgence["instClients"]
                res00 = instAgence['agenceCode']
                res01 = instAgence['dateExtraction']
                res02 = instAgence['verificationCoherence']
                
                chld =  [
                    dbc.Row(dbc.Col(id = 'idRadioReseau')),
                    html.Hr(),
                    dbc.Row(dbc.Col('Chiffres clés des Souscripteurs', className='h5')),
                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card(
                                (card_content("Primes",instClients.totalPrime,"nb")),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(
                                (card_content("Solde",instClients.totalSolde,"nb")),style={'text-align': 'center'}, color="danger", outline=True)
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card(
                                (card_content("Nombre de clients",instClients.totalNombre,"nb")),style={'text-align': 'center'}, color="danger", outline=True)
                            ],xs=12, sm=12, md=12, lg=4, xl=4),
                        ],
                    ),
                    html.Br(),

                    dbc.Row(
                        [        
                            dbc.Col([dbc.Card((
                                card_content3("Meilleur client",
                                    instClients.clientPlusRentable['nom'],
                                    instClients.clientPlusRentable['prime'],
                                    instClients.clientPlusRentable['primePercentage'],
                                    instClients.clientPlusRentable['txPaiement'],
                                    instClients.clientPlusRentable['solde'],

                                )
                                ),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card((
                                card_content3("2e Meilleur client",
                                    instClients.client2ePlusRentable['nom'],
                                    instClients.client2ePlusRentable['prime'],
                                    instClients.client2ePlusRentable['primePercentage'],
                                    instClients.client2ePlusRentable['txPaiement'],
                                    instClients.client2ePlusRentable['solde'],

                                )
                                ),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=4, xl=4),

                            dbc.Col([dbc.Card((
                                card_content3("Recouvrement client",
                                    instClients.clientPlusGrosDeficit['nom'],
                                    instClients.clientPlusGrosDeficit['prime'],
                                    instClients.clientPlusGrosDeficit['primePercentage'],
                                    instClients.clientPlusGrosDeficit['txPaiement'],
                                    instClients.clientPlusGrosDeficit['solde'],

                                )
                                ),style={'text-align': 'center'}, color="danger", outline=True )
                            ],xs=12, sm=12, md=12, lg=4, xl=4),



                        
                        ],
                    ),
                    html.Hr(),
                    dbc.Row(dbc.Col("Représentation graphique des Souscripteurs", className='h5')),
                    dbc.Row([
                        dbc.Col([
                            html.P(">>>> Sélection de l'indicateur: "),
                            dcc.Dropdown(
                                id='idDropdownClient3',
                                options=[
                                    {'label': i, 'value': i} for i in lsIndicateur
                                ],
                                optionHeight=45,
                                value = lsIndicateur[0],
                                clearable = False,
                                style={
                                    'font-size': "90%",
                                },
                            )
                        ], xs=12, sm=12, md=12, lg=4, xl=4)
                    ]),

                
                    dbc.Row(
                        dbc.Col(
                            id = 'idGraphClient3',
                            xs=12, sm=12, md=12, lg=12, xl=12  
                        ),
                    ),
                    html.Br(),
                    
                
    
                ]          
            
            return  chld, res00,res01,res02
  
    return ">>>> Sélectioner une Vue"



#Vue Assurances
@app.callback(  
    [   
        Output("idGraphAssurance3", "children"),
        
    ],  
    [
        Input("idDropdownAgences","value"),
        Input("idRadioAssurance3", "value"),
        Input("idDropdownAssurance3","value"),
        Input("idRadioReseau","value"),
 

    ]
)
def souscripteur_update(v0,v1,v2,v3):  
    if v0==0:
        if v3== None:
            v3='all'
        instAgencesAssurances = instAgences
        res1 = [
            graph_assurance_agences(instAgencesAssurances,v1,v2,v3)
        ]
    else:
        instAgence = func_instAgence(v0)
        instAssurances = instAgence["instAssurances"]
    
        res1 = [
                
                graph_assurance (instAssurances,v1,v2),
            ]
    return res1



#Vue Client
@app.callback(  
    [   
        Output("idGraphClient3", "children"),
        
    ],  
    [
        Input("idDropdownAgences","value"),
        Input("idDropdownClient3","value"),
        Input("idRadioReseau","value"),
 

    ]
)
def souscripteur_update(v0,v1,v2):  
    
    if v0==0:
        if v2== None:
            v2='all'
        instClientAgence = instAgences
        res1 = [
            
            graph_client_agence (instClientAgence,v1,v2),
        ]
    else: 
        instAgence = func_instAgence(v0)
        instClients = instAgence["instClients"]
        
        res1 = [
            
            graph_client (instClients,v1),
        ]
    return res1

# #Agences Quittances

# @app.callback(  
#     [   
#         Output("idRadioAgences", "value"),
        
#     ],  
#     [
#          Input("idRadioAgences","n_clicks"),
    
#     ]
# )
# def AgencesQuittance_update(v0):  
    
#     instAgence = func_instAgence(v0)
#     instClients = instAgence["instClients"]
    
#     res1 = [
        
#         graph_client (instClients,v1),
#     ]
#     return res1





# #Vue Agences
# @app.callback(  
#     [   
#         #quittances
#         Output("idCardNomAgence","children"),
#         Output("idDateExtraction","children"),
#         Output("idVerif","children"),
#         Output("cardQ11", "children"),
#         Output("cardQ12", "children"),
#         Output("cardQ13", "children"),
#         Output("cardQ21", "children"),
#         Output("cardQ22", "children"),
#         Output("cardQ23", "children"),
#         Output("cardQ31", "children"),
#         Output("cardQ32", "children"),
#         Output("cardQ33", "children"),
#         Output("idGraphQuitEvolPrime", "children"),
#         Output("idGraphQuitEvolSolde", "children"),
        

          
        
#     ],  
#     [
      
#         Input("idDropdownAgences","value"),
 

#     ]
# )
# def selected_agences_update(v1):  
    
#     if v1 == "ACT02":
#         instAgence = instAct
#     elif v1 == "ANALAVORY":
#         instAgence = instAnalavory

           
#     #quittances
#     instQuittances = instAgence["instQuittances"]

#     res00 = instAgence['agenceCode']
#     res01 = instAgence['dateExtraction']
#     res02 = instAgence['verif']
#     res11 = card_content("Primes",instQuittances.totalPrime,"nb")       
#     res12 = card_content("Solde",instQuittances.totalSolde,"nb")
#     res13 = card_content("Nombre de quittances",instQuittances.totalNombre,"nb")
#     res21 = card_content("Primes 2021",instQuittances.evolAnnuel2(2021,'prime'),"nb")
#     res22 = card_content("Solde 2021",instQuittances.evolAnnuel2(2021,'solde'),"nb")
#     res23 = card_content("Taux de paiement 2021",instQuittances.evolAnnuel2(2021,'tx_paiement'),"%")
#     res31 = card_content("Primes 2022 (YTD)",instQuittances.evolAnnuel2(2022,'prime'),"nb")
#     res32 = card_content("Solde 2022 (YTD)",instQuittances.evolAnnuel2(2022,'solde'),"nb")
#     res33 = card_content("Taux de paiement 2022 (YTD)",instQuittances.evolAnnuel2(2022,'tx_paiement'),"%")
#     resGraphQuittancePrime = [graph_evolQuittance_prime(instQuittances)]
#     resGraphQuittanceSolde = [graph_evolQuittance_solde(instQuittances)]

#     return res00, res01, res02, res11, res12, res13, res21, res22, res23,res31, res32, res33,resGraphQuittancePrime, resGraphQuittanceSolde

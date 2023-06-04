
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

# Dash package
from pandas.io.sql import has_table
import dash
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_html_components as html
import dash_table
import dash_core_components as dcc
#from dash import html
#from dash import dash_table
#from dash import dcc
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

from apps import nhObjet


today = datetime.date.today()
print("Today's date on orass-v2 page: ", today)


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
df_orass  = pd.read_sql("SELECT * FROM orass", dbConnection)

#Close the database connection
dbConnection.close()

print("welcome orass v2 page")

import sys
print("mem orass-v2: ", sys.getsizeof(df_orass))
print()

# Les variables
dateExtraction = df_orass['dateExtraction'].dt.date[0]
agenceCode = str(df_orass.head(1)['Code intermédiaire'][0])
agenceCode = agenceCode +'-'+ str(df_orass.head(1)['intermédiaire'][0])



# Managing df_orass
## quittance 2010010090 - type: annulation, client 112325 cient non CAR
## quittance à supprimer
df_orass = df_orass[df_orass['Num quittance'] != 2010010090]
#assureur 112056 avec un contrat annulé et en doublon avec un autre client
df_orass = df_orass[df_orass['Code souscripteur'] != 112056]

## decallage de colonne 
## quittance 2010020507 - prime totale est une date à changer et code client NaN
#y = df_orass[df_orass['Num quittance'] == 2010020507]
#idx = y.index[0]
#col = 'Code souscripteur'
#colRef = y.columns.tolist().index(col)
#while colRef < len(y.columns)-1:
#    df_orass.loc[idx,y.columns[colRef]] = y[y.columns[colRef+1]].tolist()[0]
#    colRef = colRef+1

## etat de quittance à changer
q = [2010023216,
2010023547,
2010023959,
2010024338,
2010024713,
2010025322]
for i in q:
    y = df_orass[df_orass['Num quittance'] == i]
    idx = y.index[0]
    df_orass.loc[idx,'Etat Quittance'] = " ESP."


## cols chgmt de type 
df_orass = df_orass.astype({"Prime totale quittance": float , "Code souscripteur": int })



# Les objets et les collections


## Les instances

instDf = df_orass.copy()

instQuittances = nhObjet.Quittances(instDf)
instQuittances.initCollection()
instQuittances.calculStats()

instAssurances = nhObjet.Assurances(instDf,instQuittances)
instAssurances.initCollection()
#print(instAssurances.collectionAssurances.items())
#exit()
instAssurances.calculStats()

instClients = nhObjet.Clients(instDf,instAssurances)
instClients.initCollection()
instClients.calculStats()

## verification des instances
if (instQuittances.totalSolde - instAssurances.totalSolde == 0) & (instClients.totalSolde-instAssurances.totalSolde == 0) & (instQuittances.totalPrime - instAssurances.totalPrime == 0) & (instClients.totalPrime-instAssurances.totalPrime == 0):
    verif = "Cohérence OK"
else:
    ls_c = []
    for k,c in instClients.collectionClients.items():
        for a in c.assurances:
            ls_c.append (a)
    ls_a = []
    for k,a in instAssurances.collectionAssurances.items():
        ls_a.append (k)
    dup = {x for x in ls_c if ls_c.count(x) > 1}
    verif = "Données à revérifier: Doublon de contrat " + str(dup)





# les fonctions pour la viz

lsIndicateur = [instAssurances.dfBranche.columns[i] for i in range(1,5,1)]

## vue quittance
## definition des variables à utiliser
df_graph = instQuittances.evolAnnuel
df_graph = df_graph[df_graph.index>2019]
df_graph = df_graph[df_graph.index<2023]

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

def graph_evolQuittance_prime():

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
            ticktext=["2020", "2021", "2022","2023 YTD"],
            tickvals=[2020, 2021, 2022, 2023],
        )

    dcc_graph = dcc.Graph(figure = fig)


    return dcc_graph

def graph_evolQuittance_solde():
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
            ticktext=["2020", "2021", "2022","2023 YTD"],
            tickvals=[2020, 2021, 2022, 2023],
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


def graph_assurance (assurance_type,assurance_y):

    #assurance_type = 'categorie'
    #assurance_y = 'solde'

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

def graph_client (client_y):

    #client_y = 'solde'

    df_graphclient = instClients.dfClient

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


layout = dbc.Container(
    [
        #dcc.Store(id="store"),
        #html.H1("Cabinet d'Assurance Razafindrakola - Portefeuille"),
        #html.Hr(),
        dbc.Row([
            
            dbc.Col([                  
                html.Br(),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody(
                            [
                                html.P("Agence",style={'text-align': 'center','font-weight': 'bold'},className="card-subtitle"),
                                html.P(agenceCode, style={'text-align': 'center'},className="card-text"),
                                                                
                            ]
                        ),

                    ]),

                    
                    html.Br(),
                    dbc.Card([
                        dbc.CardBody(
                            [
                                html.P("Date d'extraction",style={'text-align': 'center','font-weight': 'bold'},className="card-subtitle"),
                                html.P(dateExtraction, style={'text-align': 'center'},className="card-text"),
                                html.P('Vérification des données: ',style={ 'font-size': '10px'},className="card-subtitle"),
                                html.P(verif, style={ 'font-size': '10px'},className="card-text"),
                                
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
                    dbc.Tab(label="Vue Quittance", tab_id="idQuittance"),
                    dbc.Tab(label="Vue Assurance", tab_id="idAssurance"),
                    dbc.Tab(label="Vue Client", tab_id="idClient"),
                   

                ],
                id="id-tabs-v2",
                active_tab="None",
                ),
                html.Br(),
                html.Div(id="id-tab-content-v2")#, className="p-4"),
            ],xs=12, sm=6, md=6, lg=10, xl=10
            ),

            
        ]),
    ],fluid=True
)

#callback tab
@app.callback(
    Output("id-tab-content-v2", "children"),
    Input("id-tabs-v2", "active_tab"),
)
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab is not None:
        
        if active_tab == "idQuittance":
            
            chld = [
                #html.P(),
                dbc.Row(dbc.Col('Chiffres clés des Quittances émises', className='h5')),
                dbc.Row(
                    [        
                        dbc.Col([dbc.Card(
                            (card_content("Primes",instQuittances.totalPrime,"nb")),style={'text-align': 'center'}, color="danger", outline=True, id='card1a-id')
                        ],xs=12, sm=12, md=12, lg=4, xl=4),

                        dbc.Col([dbc.Card(
                            (card_content("Solde",instQuittances.totalSolde,"nb")),style={'text-align': 'center'}, color="danger", outline=True, id='card1b-id')
                        ],xs=12, sm=12, md=12, lg=4, xl=4),

                        dbc.Col([dbc.Card(
                            (card_content("Nombre de quittances",instQuittances.totalNombre,"nb")),style={'text-align': 'center'}, color="danger", outline=True, id='card1c-id')
                        ],xs=12, sm=12, md=12, lg=4, xl=4),
                    ],
                ),
                html.Br(),
                dbc.Row(
                    [        
                        dbc.Col([dbc.Card(
                            (card_content("Primes 2021",instQuittances.evolAnnuel2(2021,'prime'),"nb")),style={'text-align': 'center'}, color="danger", outline=True, id='card2a-id')
                        ],xs=12, sm=12, md=12, lg=4, xl=4),

                        dbc.Col([dbc.Card(
                            (card_content("Solde 2021",instQuittances.evolAnnuel2(2021,'solde'),"nb")),style={'text-align': 'center'}, color="danger", outline=True, id='card2b-id')
                        ],xs=12, sm=12, md=12, lg=4, xl=4),

                        dbc.Col([dbc.Card(
                            (card_content("Taux de paiement 2021",instQuittances.evolAnnuel2(2021,'tx_paiement'),"%")),style={'text-align': 'center'}, color="danger", outline=True, id='card2c-id')
                        ],xs=12, sm=12, md=12, lg=4, xl=4),
                    ],

                ),
                html.Br(),
                dbc.Row(
                    [        
                        dbc.Col([dbc.Card(
                            (card_content("Primes 2023 (YTD)",instQuittances.evolAnnuel2(2023,'prime'),"nb")),style={'text-align': 'center'}, color="danger", outline=True, id='card2a-id')
                        ],xs=12, sm=12, md=12, lg=4, xl=4),

                        dbc.Col([dbc.Card(
                            (card_content("Solde 2023 (YTD)",instQuittances.evolAnnuel2(2023,'solde'),"nb")),style={'text-align': 'center'}, color="danger", outline=True, id='card2b-id')
                        ],xs=12, sm=12, md=12, lg=4, xl=4),

                        dbc.Col([dbc.Card(
                            (card_content("Taux de paiement 2023 (YTD)",instQuittances.evolAnnuel2(2023,'tx_paiement'),"%")),style={'text-align': 'center'}, color="danger", outline=True, id='card2c-id')
                        ],xs=12, sm=12, md=12, lg=4, xl=4),
                    ],

                ),
                html.Hr(),
                dbc.Row(dbc.Col("Représentation graphique d'évolution des réalisations", className='h5')),
                dbc.Row(
                    [
                        dbc.Col([graph_evolQuittance_prime() 
                        ], xs=12, sm=12, md=12, lg=6, xl=6),
                        dbc.Col([graph_evolQuittance_solde(),
                        ], xs=12, sm=12, md=12, lg=6, xl=6),

                    ],
                ),


            ]
            
            
            return  chld


        elif active_tab == "idAssurance":   
            chld =  [
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
                            id = 'idRadioAssurance',
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
                            id='idDropdownAssurance',
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
                        #children = souscripteur_graph(dt_ref,list_souscripteur[0]),
                        id = 'idGraphAssurance',
                        xs=12, sm=12, md=12, lg=12, xl=12  
                    ),
                ),
                html.Br(),
                
                        
            
            ]     

                
            return  chld

        elif active_tab == "idClient":     
            chld =  [
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
                            id='idDropdownClient',
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
                        #children = souscripteur_graph(dt_ref,list_souscripteur[0]),
                        id = 'idGraphClient',
                        xs=12, sm=12, md=12, lg=12, xl=12  
                    ),
                ),
                html.Br(),
                
            
 
            ]          
            
            return  chld
  
    return ">>>> Sélectioner une Vue"



#Vue Assurances
@app.callback(  
    [   
        Output("idGraphAssurance", "children"),
        
    ],  
    [
        Input("idRadioAssurance", "value"),
        Input("idDropdownAssurance","value"),
 

    ]
)
def souscripteur_update(v1,v2):  
    res1 = [
        
        graph_assurance (v1,v2),
    ]
    return res1

#Vue Client
@app.callback(  
    [   
        Output("idGraphClient", "children"),
        
    ],  
    [
      
        Input("idDropdownClient","value"),
 

    ]
)
def souscripteur_update(v1):  
    res1 = [
        
        graph_client (v1),
    ]
    return res1

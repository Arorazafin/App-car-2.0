
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
print("Today's date orass page: ", today)


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
df_etat_quittance  = pd.read_sql("SELECT * FROM etat_quittance", dbConnection)
df_police  = pd.read_sql("SELECT * FROM police", dbConnection)
df_assurance_categorie  = pd.read_sql("SELECT * FROM assurance_categorie", dbConnection)
df_assurance_branche  = pd.read_sql("SELECT * FROM assurance_branche", dbConnection)
df_souscripteur  = pd.read_sql("SELECT * FROM souscripteur", dbConnection)
df_recouvrement = pd.read_sql("SELECT * FROM recouvrement", dbConnection)
#Close the database connection
dbConnection.close()
#test
#print('table recouvrement',df_recouvrement.head(5))

#merger les tables
df = df_etat_quittance.set_index('Num police').join(df_police.set_index('Num police'))
df = df.join(df_souscripteur.set_index('Code souscripteur'), 'Code souscripteur')
df = df.join(df_assurance_branche.set_index('Code branche'), 'Code branche')
df = df.join(df_assurance_categorie.set_index('Code catégorie'), 'Code catégorie')
df.reset_index(inplace=True)

##remove duplicate values
# sorting by first name
df.sort_values("idEtat_quittance", inplace = True)
# dropping ALL duplicate values
df.drop_duplicates(subset ="idEtat_quittance",
                     keep = "first", inplace = True)



#remove some data
#print(df.columns)
#df = df[df['date_extraction'].dt.date>datetime.date(2021,7,1)]

#date reference
dt_range = pd.to_datetime(df['date_extraction'].unique())
dt_range = list(dt_range)

#dt_ref = dt_range[len(dt_range)-1]
#t_var = dt_range[len(dt_range)-2]

dt_range.sort(reverse = True)
dt_ref = dt_range[0]
dt_var = dt_range[0]

print('Last date in orass db: ',dt_ref)
import sys
print("mem orass: ", sys.getsizeof(df))
#print(dt_var)
#print(df_recouvrement)
#print(df[df["Num quittance"] == 2010025257] )


#principals columns
col_key = ['date_extraction',
           'Libellé branche',
           'Libellé catégorie',
           'souscripteur',
           'Num quittance',
           'Num police',
           'Date effet police',
            'Date échéance police',
            'Date effet quittance',
            'Date échéance quittance',
           'Prime totale quittance',
            'Montant Encaissé quittance',
            'Date Encaiss quittance',
            'Réf Encaissement',
           'Etat Quittance',
           'etat_police']

portefeuille_list = dict()
portefeuille_list = {
            'souscripteur': ['Nombre de Souscripteurs', 'souscripteur'],
            'police' : ['Nombre de Polices','Num police'],
            'quittance' :['Nombre de Quittances','Num quittance'],
            'p_impayee' :['Prime impayée','Prime totale quittance'],
            'p_souscrite': ['Prime souscrite','Prime totale quittance'],
            'p_encaissee' : ['Prime encaissée','Montant Encaissé quittance'],
}


recouvrement_reset_list = [
    0,
    '',
    None,
    0,
    'cheque',
    0,
    0,
    '',
    '',
    'Non-Régularisé',

]

#debug
print(df.columns)
#print(df['Montant Encaissé quittance'])


#definition d'autres variables
df1 = df[col_key].copy()
#df1['Montant Encaissé quittance']=df1['Montant Encsaissé quittance'].fillna(0)
#df1['Prime_solde'] = [e-abs(p) if (t =="Non Réglée") or (t=="Réglée") else e-p for (p,e,t) in zip(df1['Prime totale quittance'],df1['Montant Encaissé quittance'],df1['Etat Quittance'] )]
df1['Date_echeance_mensuelle']  =  [float('nan') if math.isnan(dt.year) else datetime.date(dt.year,dt.month,1) for dt in df1['Date échéance quittance']]
#df1['Prime_impayee_envie'] = [ p if e=='enVie' else 0 for (e,p) in zip (df1['etat_police'],df1['Prime_solde'])]
df1['Prime_impayee_envie'] = [ -1*p if (e=='enVie') & (q=='Impayée') else 0 for (e,p,q) in zip (df1['etat_police'],df1['Prime totale quittance'],df1['Etat Quittance'])] 
df1['Prime_payee_envie'] = [ p if e=='enVie' else 0 for (e,p) in zip (df1['etat_police'],df1['Montant Encaissé quittance'])]
df1['Prime_impayee_echue'] = [ -1*p if (e=='echue') & (q=='Impayée') else 0 for (e,p,q) in zip (df1['etat_police'],df1['Prime totale quittance'],df1['Etat Quittance'])]
df1['Prime_payee_echue'] = [ p if e=='echue' else 0 for (e,p) in zip (df1['etat_police'],df1['Montant Encaissé quittance'])]
df1['Prime_non_reglee'] = [ p if q=='Non Réglée' else 0 for (q,p) in zip (df1['Etat Quittance'],df1['Prime totale quittance'])]

#dubug
#print(df1.head(10)) 
#print(df_souscripteur.head(10))
#renouvelement par client et trié
dfxx = df1[df1['date_extraction']== datetime.datetime(2023,5,11,0,0)]
print(datetime.datetime(2023,5,11,0,0)) 
dfxx = dfxx[dfxx['Date_echeance_mensuelle']==datetime.date(2023,6,1)]
dfxx = dfxx.groupby('souscripteur')['Prime totale quittance'].sum()
dfxx= dfxx.sort_values(ascending=False)
print(dfxx.head(10))




#list dropdown
list_etatC = ['All','enVie','echue']

list_mois = pd.DataFrame(df1['Date_echeance_mensuelle'].unique()).copy()
list_mois.dropna(inplace = True)
list_mois = list(list_mois.values)
list_mois = [l[0] for l in list_mois]
list_mois.sort()
list_mois.insert(0,'All')


list_souscripteur = pd.DataFrame(df_souscripteur['souscripteur'].unique()).astype(str)
list_souscripteur.sort_values(0, axis =0, inplace = True)
list_souscripteur = list(list_souscripteur.values)
list_souscripteur = [l[0] for l in list_souscripteur]

list_prime_encaisse =['nombre', 'prime', 'encaisse']

list_num_police =  pd.DataFrame(df_police['Num police'].unique()).astype(int)
list_num_police.sort_values(0, axis =0, inplace = True)
list_num_police = list(list_num_police.values)
list_num_police = [l[0] for l in list_num_police]

#list etat quittance
list_etat_quittance = pd.DataFrame(df1['Etat Quittance'].unique())
list_etat_quittance.dropna(inplace = True)
list_etat_quittance = list(list_etat_quittance.values)
list_etat_quittance = [l[0] for l in list_etat_quittance]
list_etat_quittance.sort()
list_etat_quittance.insert(0,'All')


def portfolio_dynamic(p):
    
    
    if p == 'p_impayee': 
        df_temp = df1[df1['Etat Quittance']== 'Impayée']
        x = df_temp.groupby('date_extraction')[portefeuille_list[p][1]].sum()
    elif p == 'p_souscrite':
        x =  df1.groupby('date_extraction')[portefeuille_list[p][1]].sum()
    elif p == 'p_encaissee':
        x = df1.groupby('date_extraction')[portefeuille_list[p][1]].sum()
    elif p == 'police':   
        x = df1.groupby('date_extraction')[portefeuille_list[p][1]].nunique()
    elif p == 'souscripteur':   
        x = df1.groupby('date_extraction')[portefeuille_list[p][1]].nunique()
    elif p == 'quittance':   
        x = df1.groupby('date_extraction')[portefeuille_list[p][1]].nunique()
    
    x = pd.DataFrame(x)
    x = x.join(x.shift(),rsuffix='_1')
    x['Variation'] = x.iloc[:,0]-x.iloc[:,1]
    x.fillna(0, inplace=True)
    return x

def portfolio_dynamic_graph(p):
    
    fig = go.Figure()

    dynamique1 = portfolio_dynamic(p)
    fig.add_trace(go.Bar(x=dynamique1.index,y=dynamique1[portefeuille_list[p][1]]))

    fig.update_layout( title_text= portefeuille_list[p][0])
    fig.update_layout(title_x=0.5)
    dcc_graph = dcc.Graph(figure = fig)

    return dcc_graph


def portfolio_dynamic_toDate(p,dt,dt_var):
    x = portfolio_dynamic(p)
    x1 = x.loc[dt,:][0]
    v = x.loc[dt_var:dt,'Variation'].sum()
    #dv = x.index[len(x)-2]
    #xVar = x.loc[dt_var,:][0]
    #v = x1-xVar
    diff_date = np.busday_count( dt_var.date(), dt.date(), weekmask =[1,1,1,1,1,0,0] ) + 1
    #moy = v / diff_date.days
    moy = v / diff_date
    return x1, v, diff_date, moy

def new_quittance(dt,dt_var,eq):

    i = dt_range.index(dt_var)
    dt_var = dt_range[i+1]

    l1 = df1[df1['date_extraction'] == dt]['Num quittance'].unique()
    l2 = df1[df1['date_extraction'] == dt_var]['Num quittance'].unique()
    l3 = list(set(l1) ^ (set(l2)))
    dfx = df1[df1['Num quittance'].isin(l3)]
    dfx.drop_duplicates(subset=['Num police'], inplace=True, keep='last')

    if eq != 'All':
        dfx = dfx[dfx['Etat Quittance']==eq].copy()

        
    sum_impayee = dfx[dfx['Etat Quittance']=='Impayée']['Prime totale quittance'].sum() 
    sum_impayee = -1*sum_impayee
    sum_payee = dfx[dfx['Etat Quittance']=='Payée']['Montant Encaissé quittance'].sum() 
    sum_ristourne = dfx[dfx['Etat Quittance']=='Non Réglée']['Prime totale quittance'].sum()
    sum_prime =  dfx['Prime totale quittance'].sum()
    poids_impaye = (sum_impayee /sum_prime)*100
    solde = sum_prime + sum_impayee

    res = dict()
    res ={
            'df' : dfx,
            'sum_prime': sum_prime,
            'sum_ristourne': sum_ristourne,
            'sum_impayee': sum_impayee,
            'poids_impaye' : poids_impaye,
            'sum_payee': sum_payee,
            'solde': solde,

    }

    return res



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









def card_content(p,dt,dt_var):

    res = [
        dbc.CardHeader(portefeuille_list[p][0]),
        dbc.CardBody(
            [
                html.H5('{:,.0f}'.format(portfolio_dynamic_toDate(p,dt,dt_var)[0]), className="card-title"),
                #html.P(str(round(portfolio_dynamic_toDate(p,dt,dt_var)[1]))+" (Variation au " + str(portfolio_dynamic_toDate(p,dt,dt_var)[2].date())+")"
                #, className="card-text",),
                html.P(str('{:,.0f}'.format(portfolio_dynamic_toDate(p,dt,dt_var)[1]))+" (Variation sur " + str(portfolio_dynamic_toDate(p,dt,dt_var)[2]) + " jours)", className="card-text"),
                html.P(str('{:,.1f}'.format(portfolio_dynamic_toDate(p,dt,dt_var)[3],1))+" (moyenne par jour)" ),
                
                
            ]
        ),
    ]

 
    return res

def renouvellement_table(ref_date,etat_q, mois):

    
    
    
    dfx = df1[df1['date_extraction']== ref_date]
    
    
    if (etat_q == 'All') & (mois != 'All'):
        mois = datetime.datetime.strptime(mois, '%Y-%m-%d')
        mois = datetime.datetime.date(mois)
       
        dfx = dfx[dfx['Date_echeance_mensuelle']==mois]
    elif (etat_q != 'All') & (mois == 'All'):
        dfx = dfx[dfx['Etat Quittance']==etat_q]
    elif (etat_q != 'All') & (mois != 'All'):
        mois = datetime.datetime.strptime(mois, '%Y-%m-%d')
        mois = datetime.datetime.date(mois)
        dfx = dfx[dfx['Etat Quittance']==etat_q]
        dfx = dfx[dfx['Date_echeance_mensuelle']==mois]       

    sum_impayee = dfx[dfx['Etat Quittance']=='Impayée']['Prime totale quittance'].sum() 
    sum_impayee = -1*sum_impayee
    sum_payee = dfx[dfx['Etat Quittance']=='Payée']['Montant Encaissé quittance'].sum() 
    sum_ristourne = dfx[dfx['Etat Quittance']=='Non Réglée']['Prime totale quittance'].sum()
    sum_prime =  dfx['Prime totale quittance'].sum()
    poids_impaye = (sum_impayee /sum_prime)*100
    solde = sum_prime + sum_impayee

    res = dict()
    res ={
            'df' : dfx,
            'sum_prime': sum_prime,
            'sum_ristourne': sum_ristourne,
            'sum_impayee': sum_impayee,
            'poids_impaye' : poids_impaye,
            'sum_payee': sum_payee,
            'solde': solde,

    }
    
    return res


def renouvellement_graph(ref_date, etat, mois, zoom):
    
    df_graph = df1[df1['date_extraction']== ref_date]    
    
    
    if (etat == 'All') & (mois == 'All'):
        df_graph = df_graph.copy()
    elif etat == 'All':
        df_graph = df_graph[df_graph['Date_echeance_mensuelle']==mois]
    elif mois == 'All':    
        df_graph = df_graph[df_graph['etat_police']==etat]       
    else:
        df_graph = df_graph[df_graph['Date_echeance_mensuelle']==mois]
        df_graph = df_graph[df_graph['etat_police']==etat]
        
    
 
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_graph['Date_echeance_mensuelle'],
                         y=df_graph['Prime_payee_envie'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime payée enVie",
                        )
    )
    
    fig.add_trace(go.Bar(x=df_graph['Date_echeance_mensuelle'],
                         y=df_graph['Prime_impayee_envie'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime impayée enVie",
                        )
    )
    
    fig.add_trace(go.Bar(x=df_graph['Date_echeance_mensuelle'],
                         y=df_graph['Prime_payee_echue'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime payée échue",
                        )
    )
    fig.add_trace(go.Bar(x=df_graph['Date_echeance_mensuelle'],
                         y=df_graph['Prime_impayee_echue'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime impayée échue",
                        )
    )
    
    x1 = datetime.date(2020,1,1)
    if zoom =='25%':
        x2 = datetime.date(2021,12,1)
    elif zoom=='50%':
        x2 = datetime.date(2023,12,1)
    elif zoom=='75%':
        x2 = datetime.date(2025,12,1)
    elif zoom=='100%':
        x2 = datetime.date(2030,12,1)
    fig.update_xaxes(range=[x1,x2])

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

def impayees_graph (ref_date, zoom):
    
    df_impaye = df1[(df1['Etat Quittance'] == "Impayée") | (df1['Etat Quittance'] == "Non Reglée" )] 
    df_impaye = df_impaye[df_impaye['date_extraction'] == ref_date] 

    df_impaye = df_impaye.sort_values('Prime totale quittance', axis = 0, ascending=False)
    
    df_graph = df_impaye.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_graph['souscripteur'].astype(str),
                         y=df_graph['Prime totale quittance'],
                         hovertext=df_graph['etat_police'],
                        )
                 )
    
    if zoom =='25%':
        x2 = 50
    elif zoom=='50%':
        x2 = 100
    elif zoom=='75%':
        x2 = 500
    elif zoom=='100%':
        x2 = 1000
    fig.update_xaxes(range=[0,x2])
    
    dcc_graph = dcc.Graph(figure = fig)

    return dcc_graph
    

# souscripteur/police
def souscripteurPolice_graph(dt,s_p,s,p):
    dfx = df1[df1['date_extraction']== dt] 
    
    if s_p == 'souscripteur':   
        df_graph = dfx[dfx['souscripteur']==s]
    elif s_p == 'police':
        df_graph = dfx[dfx['Num police']==p]

    #debug
    print(dfx.iloc[:10,:6])    
    print(df_graph.head(10)) 

    sum_impayee = df_graph[df_graph['Etat Quittance']=='Impayée']['Prime totale quittance'].sum() 
    sum_impayee = -1*sum_impayee
    sum_payee = df_graph[df_graph['Etat Quittance']=='Payée']['Montant Encaissé quittance'].sum() 
    sum_ristourne = df_graph[df_graph['Etat Quittance']=='Non Réglée']['Prime totale quittance'].sum()
    sum_prime =  df_graph['Prime totale quittance'].sum()
    poids_impaye = (sum_impayee /sum_prime)*100
    solde = sum_prime + sum_impayee
        
    fig = go.Figure()
    
    fig.add_trace(go.Bar( 
                         x=df_graph['Num police'].astype(str),
                         y=df_graph['Prime_payee_envie'],
                         hovertext=df_graph['Etat Quittance'],
                         name="Prime payée enVie",
                        )
    )
    
    fig.add_trace(go.Bar( 
                         x=df_graph['Num police'].astype(str),
                         y=df_graph['Prime_impayee_envie'],
                         hovertext=df_graph['Etat Quittance'],
                         name="Prime impayée enVie"
                        )
    )
    
    fig.add_trace(go.Bar(x=df_graph['Num police'].astype(str),
                         y=df_graph['Prime_payee_echue'],
                         hovertext=df_graph['Etat Quittance'],
                         name="Prime payée échue",
                        )
    )
    
    fig.add_trace(go.Bar(x=df_graph['Num police'].astype(str),
                          y=df_graph['Prime_impayee_echue'],
                         hovertext=df_graph['Etat Quittance'],
                         name="Prime impayée échue",
                        )
    )
    fig.add_trace(go.Bar(x=df_graph['Num police'].astype(str),
                          y=df_graph['Prime_non_reglee'],
                         hovertext=df_graph['Etat Quittance'],
                         name="Prime non reglée (ristourne)",
                        )
    )

    
    fig.update_xaxes(
        showgrid=True,
        ticks="outside",
        tickson="boundaries",
        ticklen=20
    )

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="right",
            x=1
        )
    )
    fig.update_layout(title_text=s)
    fig.update_layout(title_x=0.5)
    


    dcc_graph = dcc.Graph(figure = fig)

    res = dict()
    res ={
            'dcc_graph' : dcc_graph,
            'sum_prime': sum_prime,
            'sum_ristourne': sum_ristourne,
            'sum_impayee': sum_impayee,
            'poids_impaye' : poids_impaye,
            'sum_payee': sum_payee,
            'solde': solde,

    }

    return res

def souscripteurPolice_table(dt,s_p,s,p):

    dfx = df1[df1['date_extraction']== dt]

    if s_p == 'souscripteur':   
        dfx = dfx[dfx['souscripteur']==s]
    elif s_p == 'police':
        dfx = dfx[dfx['Num police']==p]
    
    
    dfx = dfx[col_key]
    
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

#souscripteur

def souscripteur_graph(dt,s):
       
    dfx = df1[df1['date_extraction']== dt] 
    
    if (s == 'All'):
        df_graph = dfx.copy()
    else:
        df_graph = dfx[dfx['souscripteur']==s]
        
    fig = go.Figure()
    
    fig.add_trace(go.Bar( 
                         x=df_graph['Num police'].astype(str),
                         y=df_graph['Prime_payee_envie'],
                         hovertext=df_graph['Etat Quittance'],
                         name="Prime payée enVie",
                        )
    )
    
    fig.add_trace(go.Bar( 
                         x=df_graph['Num police'].astype(str),
                         y=df_graph['Prime_impayee_envie'],
                         hovertext=df_graph['Etat Quittance'],
                         name="Prime impayée enVie"
                        )
    )
    
    fig.add_trace(go.Bar(x=df_graph['Num police'].astype(str),
                         y=df_graph['Prime_payee_echue'],
                         hovertext=df_graph['Etat Quittance'],
                         name="Prime payée échue",
                        )
    )
    
    fig.add_trace(go.Bar(x=df_graph['Num police'].astype(str),
                          y=df_graph['Prime_impayee_echue'],
                         hovertext=df_graph['Etat Quittance'],
                         name="Prime impayée échue",
                        )
    )

    
    fig.update_xaxes(
        showgrid=True,
        ticks="outside",
        tickson="boundaries",
        ticklen=20
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

def souscripteur_table(dt,s):

    dfx = df1[df1['date_extraction']== dt] 
    dfx = dfx[dfx['souscripteur']==s]
    dfx = dfx[col_key]
    
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



# details

details = dict()
details_list = {

            'nombre' :['Nombre de Quittances',''],
            'p_impayee' :['Prime impayée','Prime totale quittance'],
            'p_souscrite': ['Prime souscrite','Prime totale quittance'],
            'p_encaissee' : ['Prime encaissée','Montant Encaissé quittance'],
}

details_list2 = [details_list[k][0] for k in details_list]

details_list3 = dict()
details_list3 = {

            'Nombre de Quittances':'nombre' ,
            'Prime impayée':'p_impayee' ,
            'Prime souscrite':'p_souscrite',
            'Prime encaissée':'p_encaissee' ,
}


def portfolio_details_table(b_c,d, ref_date):
            
    dfx = df1[df1['date_extraction']== ref_date]
    if b_c == 'branche':
        col_b_c = 'Libellé branche'
    else:
        col_b_c = 'Libellé catégorie'
    
    if d == 'p_impayee': 
        df_temp = dfx[dfx['Etat Quittance']== 'Impayée']
        dfx1 = pd.crosstab(dfx[col_b_c], dfx['etat_police'],values =dfx[details_list[d][1]],aggfunc = sum ,
                        margins = True, margins_name = 'Total')
    elif d=='nombre':
        dfx1 = pd.crosstab(dfx[col_b_c], dfx['etat_police'], margins = True, margins_name = 'Total')
    else:
        dfx1 = pd.crosstab(dfx[col_b_c], dfx['etat_police'],values =dfx[details_list[d][1]],aggfunc = sum ,
                        margins = True, margins_name = 'Total')
    
    x = pd.DataFrame(dfx1)
    x.fillna(0, inplace = True)
    x = x.round(0)
    x.reset_index(inplace=True)

    dcc_table = dash_table.DataTable(
        #id='table',
        columns=[{"name": i, "id": i} for i in x.columns],
        data=x.to_dict('records'),
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
        fixed_rows={'headers': True},


    )
    return dcc_table

#categorie
def portfolio_details_catGraph(ref_date):
     
    df_graph = df1[df1['date_extraction']== ref_date]    
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_graph['Libellé catégorie'],
                         y=df_graph['Prime_payee_envie'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime payée enVie",
                        )
    )
    
    fig.add_trace(go.Bar(x=df_graph['Libellé catégorie'],
                         y=df_graph['Prime_impayee_envie'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime impayée enVie",
                        )
    )
    
    fig.add_trace(go.Bar(x=df_graph['Libellé catégorie'],
                         y=df_graph['Prime_payee_echue'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime payée échue",
                        )
    )
    fig.add_trace(go.Bar(x=df_graph['Libellé catégorie'],
                         y=df_graph['Prime_impayee_echue'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime impayée échue",
                        )
    )

    fig.update_layout(title_text='Assurance par Branche')
    fig.update_layout(title_x=0.5)
    

      
    


    dcc_graph = dcc.Graph(figure = fig)

    return dcc_graph


#Branche
def portfolio_details_braGraph(ref_date):
    
    df_graph = df1[df1['date_extraction']== ref_date]    
        
    
 
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_graph['Libellé branche'],
                         y=df_graph['Prime_payee_envie'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime payée enVie",
                        )
    )
    
    fig.add_trace(go.Bar(x=df_graph['Libellé branche'],
                         y=df_graph['Prime_impayee_envie'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime impayée enVie",
                        )
    )
    
    fig.add_trace(go.Bar(x=df_graph['Libellé branche'],
                         y=df_graph['Prime_payee_echue'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime payée échue",
                        )
    )
    fig.add_trace(go.Bar(x=df_graph['Libellé branche'],
                         y=df_graph['Prime_impayee_echue'],
                         hovertext=df_graph['souscripteur'],
                         name="Prime impayée échue",
                        )
    )
    fig.update_layout(title_text='Assurance par Catégorie')
    fig.update_layout(title_x=0.5)
    
    

    dcc_graph = dcc.Graph(figure = fig)

    return dcc_graph




def portfolio_details_graph2(ref_dt,var_dt, branche_categorie, prime_nombre):
    
    df_graph_ref = df1[df1['date_extraction']== ref_dt]
    df_graph_var = df1[df1['date_extraction']== var_dt]
    
    #definition des colonnes
    if branche_categorie == 'branche':
        b_c = 'Libellé branche'
        title = 'Assurance par branche'
    elif branche_categorie == 'catégorie':
        b_c = 'Libellé catégorie'
        title = 'Assurance par catégorie'
    
    if prime_nombre =='prime':
        p = 'Prime totale quittance'
        df_graph_ref1 = df_graph_ref.groupby([b_c,'etat_police','Etat Quittance'])[p].sum()
        df_graph_var1 = df_graph_var.groupby([b_c,'etat_police','Etat Quittance'])[p].sum()
    elif prime_nombre =='encaisse':
        p = 'Montant Encaissé quittance'
        df_graph_ref1 = df_graph_ref.groupby([b_c,'etat_police','Etat Quittance'])[p].sum()
        df_graph_var1 = df_graph_var.groupby([b_c,'etat_police','Etat Quittance'])[p].sum()
    elif prime_nombre =='nombre':
        p = 'Num police'
        df_graph_ref1 = df_graph_ref.groupby([b_c,'etat_police','Etat Quittance'])[p].count()
        df_graph_var1 = df_graph_var.groupby([b_c,'etat_police','Etat Quittance'])[p].count()

 
    
    df_graph_diff = df_graph_ref1 - df_graph_var1
    
    df_graph_ref1 = pd.DataFrame(df_graph_ref1)
    df_graph_diff = pd.DataFrame(df_graph_diff)
    
    
    df_graph_ref1.reset_index(inplace = True)
    df_graph_ref1['hover'] = [ x+','+y for (x,y) in zip(df_graph_ref1['etat_police'],df_graph_ref1['Etat Quittance'])]
    
    
    df_graph_diff.reset_index(inplace = True)
    df_graph_diff['hover'] = [ x+','+y for (x,y) in zip(df_graph_diff['etat_police'],df_graph_diff['Etat Quittance'])]

    
    
    
    fig = make_subplots(rows=1, cols=2)
    
    
 
    #fig = go.Figure()
    fig.add_trace(go.Bar(y=df_graph_ref1[b_c],
                         x=df_graph_ref1[p],
                         text= df_graph_ref1['hover'],
                         #name="Prime payée enVie",
                         orientation='h'
                        ),
                  row=1, col=1,
                  
                 )
    
    
    fig.add_trace(go.Bar(y=df_graph_diff[b_c],
                         x=df_graph_diff[p],
                         text= df_graph_diff['hover'],
                         #name="Prime payée enVie",
                         orientation='h'
                        ),
                  row=1, col=2,
                  
                )
    
    

    

    fig.update_layout( title_text= title)
    fig.update_layout(title_x=0.5)
    fig.update_layout(showlegend=False)
   


    dcc_graph = dcc.Graph(figure = fig)

    return dcc_graph


def portfolio_details_graph3(ref_dt,var_dt, branche_categorie, prime_nombre, level_var):
    
    df_graph_ref = df1[df1['date_extraction']== ref_dt]
    df_graph_var = df1[df1['date_extraction']== var_dt]
    
    #definition des colonnes
    if branche_categorie == 'branche':
        b_c = 'Libellé branche'
        title = 'Assurance par branche'
    elif branche_categorie == 'catégorie':
        b_c = 'Libellé catégorie'
        title = 'Assurance par catégorie'
    
    if prime_nombre =='prime':
        p = 'Prime totale quittance'
        df_graph_ref1 = df_graph_ref.groupby([b_c,'etat_police','Etat Quittance'])[p].sum()
        df_graph_var1 = df_graph_var.groupby([b_c,'etat_police','Etat Quittance'])[p].sum()
    elif prime_nombre =='encaisse':
        p = 'Montant Encaissé quittance'
        df_graph_ref1 = df_graph_ref.groupby([b_c,'etat_police','Etat Quittance'])[p].sum()
        df_graph_var1 = df_graph_var.groupby([b_c,'etat_police','Etat Quittance'])[p].sum()
    elif prime_nombre =='nombre':
        p = 'Num police'
        df_graph_ref1 = df_graph_ref.groupby([b_c,'etat_police','Etat Quittance'])[p].count()
        df_graph_var1 = df_graph_var.groupby([b_c,'etat_police','Etat Quittance'])[p].count()

 
    
    df_graph_diff = df_graph_ref1 - df_graph_var1
    
    df_graph_ref1 = pd.DataFrame(df_graph_ref1)
    df_graph_diff = pd.DataFrame(df_graph_diff)
    
    
    df_graph_ref1.reset_index(inplace = True)
    df_graph_ref1['hover'] = [ x+','+y for (x,y) in zip(df_graph_ref1['etat_police'],df_graph_ref1['Etat Quittance'])]
    
    
    df_graph_diff.reset_index(inplace = True)
    df_graph_diff['hover'] = [ x+','+y for (x,y) in zip(df_graph_diff['etat_police'],df_graph_diff['Etat Quittance'])]

    
 
    fig = go.Figure()

    if level_var == 'level':
        fig.add_trace(go.Bar(y=df_graph_ref1[b_c],
                            x=df_graph_ref1[p],
                            text= df_graph_ref1['hover'],
                            #name="Prime payée enVie",
                            orientation='h'
                            ),
                    
                    )
        fig.update_layout( title_text= title)
    elif level_var == 'var':
        fig.add_trace(go.Bar(y=df_graph_diff[b_c],
                            x=df_graph_diff[p],
                            text= df_graph_diff['hover'],
                            #name="Prime payée enVie",
                            orientation='h',
                            marker_color='red'
                            ),
        
                  
                )
        fig.update_layout( title_text= title+' (variation)')
    
    

    

    
    fig.update_layout(title_x=0.5)
    fig.update_layout(showlegend=False)
   


    dcc_graph = dcc.Graph(figure = fig)

    return dcc_graph


#Recouvrement - insert
def recouvrement_insert(s,pri,gaT,gaM,acc,accRef,com,sit):
   
    query = f"""
                INSERT INTO recouvrement (
                        date_saisie, 
                        souscripteur_nom, 
                        prime_montant, 
                        garantie_type, 
                        garantie_montant, 
                        account_montant,
                        account_reference,
                        commentaire,
                        recouvrement_situation)
                    VALUES('{today}','{s}', {pri}, '{gaT}', {gaM}, {acc}, '{accRef}', '{com}','{sit}') 
                    RETURNING id;
    
             """

    conn = psycopg2.connect(dbname=database, user=user,
                            host=host, password=pwd)
    c = conn.cursor()
    c.execute(query)
    conn.commit()
    conn.close()
    
    return 'Insertion OK !'

#Recouvrement - update
def recouvrement_update(id,dt,s,pri,gaT,gaM,acc,accRef,com,sit):
   
    query = f"""
                UPDATE recouvrement 
                    SET date_saisie = '{today}', 
                        souscripteur_nom= '{s}', 
                        prime_montant= {pri}, 
                        garantie_type= '{gaT}', 
                        garantie_montant= {gaM}, 
                        account_montant= {acc},
                        account_reference= '{accRef}',
                        commentaire= '{com}',
                        recouvrement_situation= '{sit}'
                    WHERE id = {id} 
                    ;
    
             """

    conn = psycopg2.connect(dbname=database, user=user,
                            host=host, password=pwd)
    c = conn.cursor()
    c.execute(query)
    conn.commit()
    conn.close()
    
    return 'MàJ OK !'


#Recouvrement list
def recouvrement_list():
    # Connect to PostgreSQL server
    dbConnection = alchemyEngine.connect()
    # Read data from PostgreSQL database table and load into a DataFrame instance
    dfx  = pd.read_sql("SELECT * FROM recouvrement", dbConnection)
    dbConnection.close()
    return dfx

#Recouvrement select
def recouvrement_select(i):
   
    query = f""" SELECT * FROM recouvrement
                    WHERE id = {i} ;
             """

    conn = psycopg2.connect(dbname=database, user=user,
                            host=host, password=pwd)
    c = conn.cursor()
    c.execute(query)
    res_select = c.fetchall()
    conn.commit()
    conn.close()
    
    return res_select

# prime nette
def prime_nette(s):

    dfx = df1[df1['date_extraction']== dt_ref]

    dfx = dfx[dfx['souscripteur']==s]
    #dfx = dfx[dfx['Num police']==p]

    sum_impayee = dfx[dfx['Etat Quittance']=='Impayée']['Prime totale quittance'].sum() 
    #sum_impayee = -1*sum_impayee
    #sum_payee = dfx[dfx['Etat Quittance']=='Payée']['Montant Encaissé quittance'].sum() 
    sum_ristourne = dfx[dfx['Etat Quittance']=='Non Réglée']['Prime totale quittance'].sum()
    #sum_prime =  dfx['Prime totale quittance'].sum()
    #poids_impaye = (sum_impayee /sum_prime)*100
    solde =  sum_ristourne - sum_impayee


    return solde

layout = dbc.Container(
    [
        #dcc.Store(id="store"),
        #html.H1("Cabinet d'Assurance Razafindrakola - Portefeuille"),
        #html.Hr(),
        dbc.Row([
            dbc.Col([                  
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody(
                            
                            [
                                html.P('Référence'),
                                dcc.Dropdown(
                                    id='dd-dt-ref',
                                    options=[
                                        {'label': i.date(), 'value': i} for i in dt_range
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
                                    id='dd-dt-var',
                                    options=[
                                        {'label': i.date(), 'value': i} for i in dt_range
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
                    dbc.Tab(label="Portefeuille", tab_id="portefeuille_id"),
                    dbc.Tab(label="Details", tab_id="details-id"),
                    dbc.Tab(label="Renouvellement", tab_id="renouvellement_id"),
                    dbc.Tab(label="Impayées", tab_id="impayees_id"),
                    dbc.Tab(label="Souscripteurs", tab_id="souscripteur_id"),
                    dbc.Tab(label="Recouvrement", tab_id="recouvrement-id"),


                ],
                id="tabs",
                active_tab="None",
                ),
                html.Br(),
                html.Div(id="tab-content")#, className="p-4"),
            ],xs=12, sm=6, md=6, lg=10, xl=10
            ),

            
        ]),
    ],fluid=True
)


#callback tab
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab is not None:
        
        if active_tab == "portefeuille_id":
            
            chld = [
                html.P(">>>>> Page Depréciée <<<<<"),
                # dbc.Row(dbc.Col('En chiffres', className='h4')),
                # dbc.Row(
                #     [
                #         dbc.Col([dbc.Card(
                #             (card_content('souscripteur',dt_ref,dt_var)),style={'text-align': 'center'}, color="danger", outline=True, id='card-souscripteur-id')
                #         ],xs=12, sm=12, md=12, lg=4, xl=4),
                #         dbc.Col([dbc.Card(
                #             (card_content('police',dt_ref,dt_var)),style={'text-align': 'center'}, color="danger", outline=True,id='card-police-id')
                #         ],xs=12, sm=12, md=12, lg=4, xl=4),
                #         dbc.Col([dbc.Card(
                #             (card_content('quittance',dt_ref,dt_var)),style={'text-align': 'center'}, color="danger", outline=True,id='card-quittance-id')
                #         ],xs=12, sm=12, md=12, lg=4, xl=4),
                #     ],
                # ),
                # html.Br(),
                # dbc.Row(
                #     [
                #         dbc.Col([dbc.Card(
                #             (card_content('p_souscrite',dt_ref,dt_var)), style={'text-align': 'center'},color="danger", outline=True, id='card-p_souscrite-id')
                #         ],xs=12, sm=12, md=12, lg=4, xl=4),
                #         dbc.Col([dbc.Card(
                #             (card_content('p_encaissee',dt_ref,dt_var)), style={'text-align': 'center'},color="danger", outline=True,id='card-p_encaissee-id')
                #         ],xs=12, sm=12, md=12, lg=4, xl=4),
                #         dbc.Col([dbc.Card(
                #             (card_content('p_impayee',dt_ref,dt_var)),style={'text-align': 'center'}, color="danger", outline=True,id='card-p_impayee-id')
                #         ],xs=12, sm=12, md=12, lg=4, xl=4),
                #     ],
                # ),
                # html.Hr(),
                # dbc.Row(dbc.Col('Evolution', className='h4')),
                # dbc.Row(
                #     [
                #         dbc.Col([portfolio_dynamic_graph("p_souscrite"), 
                #         ], xs=12, sm=12, md=12, lg=4, xl=4),
                #         dbc.Col([portfolio_dynamic_graph("p_encaissee"),
                #         ], xs=12, sm=12, md=12, lg=4, xl=4),
                #         dbc.Col([portfolio_dynamic_graph("p_impayee"),
                #         ], xs=12, sm=12, md=12, lg=4, xl=4),
                #     ],
                # ),
                # html.Hr(),
                # dbc.Row(dbc.Col('Liste', className='h4')),
                # dbc.Row(
                #     dbc.Col([
                #         html.H6('Filtre par "état de quittance":'),
                #         dcc.Dropdown(
                #                     id='dd-etat-quittance',
                #                     options=[
                #                         {'label': i, 'value': i} for i in list_etat_quittance
                #                     ],
                #                     #optionHeight=45,
                #                     value='All',
                #                     clearable = False,
                #                     #style={
                #                     #    'font-size': "90%",
                #                     #    },
                #                 )

                #     ],xs=12, sm=12, md=12, lg=4, xl=4),
                # ),
                
                # dbc.Row(
                #     dbc.Col(
                #         id = 'new-quittance-id',
                #         xs=12, sm=12, md=12, lg=12, xl=12)
                # )
            ]
            
            
            return  chld


        elif active_tab == "renouvellement_id":   
            chld =  [
                        
                            dbc.Row(
                                dbc.Col(
                                    "Suivi des contrats d'assurance arrivés à échéance en vue de leur renouvellement",
                                    className = 'h6',
                                    xs=12, sm=12, md=12, lg=12, xl=12
                                )
                            ),
                            html.Br(),
                            dbc.Row( 
                                dbc.Col(
                                    #renouvellement_graph(dt_ref, 'All', 'All','50%'),
                                    id = 'renouvellement-graph-id',
                                    xs=12, sm=12, md=12, lg=12, xl=12
                                )
                            ),
                            html.Hr(),
                            dbc.Row(dbc.Col(
                                        "Liste", 
                                        className= 'h5',
                                        xs=12, sm=12, md=12, lg=12, xl=12
                                    )
                            ),
                            html.Br(),
                            dbc.Row([ 
                                dbc.Col([
                                    html.H6('Filtre par "etat de quittance"'),
                                    dcc.Dropdown(
                                                id='dd-etat-quittance',
                                                options=[
                                                    {'label': i, 'value': i} for i in list_etat_quittance
                                                ],
                                                optionHeight=45,
                                                value='All',
                                                clearable = False,
                                                style={
                                                    'font-size': "90%",
                                                    },
                                            )

                                ],xs=12, sm=12, md=12, lg=3, xl=3),
                                dbc.Col([
                                    html.H6('Filtre par "mois"'),
                                    dcc.Dropdown(
                                                id='dd-mois',
                                                options=[
                                                    {'label': i, 'value': i} for i in list_mois
                                                ],
                                                optionHeight=45,
                                                value=datetime.date(2021,6,1),
                                                clearable = False,
                                                style={
                                                    'font-size': "90%",
                                                    },
                                            )

                                ],xs=12, sm=12, md=12, lg=3, xl=3),


                            ]),
                            html.Br(),
                            dbc.Row(dbc.Col(  
                                        id = 'renouvellement-table-id',
                                        xs=12, sm=12, md=12, lg=12, xl=12
                                    )
                            )
            ]     

                
            return  chld

        elif active_tab == "impayees_id":     
            chld =  [
                        dbc.Row(
                            dbc.Col(
                                "Suivi des impayés pour une meilleure action de recouvrement", 
                                className='h6', 
                                xs=12, sm=12, md=12, lg=12, xl=12
                            )
                        ),
                        html.Br(),
                        dbc.Row(
                            dbc.Col(
                                id = 'impayees-graph-id',
                                xs=12, sm=12, md=12, lg=12, xl=12
                            )
                        )
                    ]          
            return  chld

        elif active_tab == "souscripteur_id":
            chld =  [
                dbc.Row([
                    dbc.Col([
                        dcc.RadioItems(
                            id = 'radio-souscripteurPolice-id',
                            options=[
                                    {'label': i, 'value': i} for i in ['souscripteur', 'police']
                            ],
                            value='souscripteur',
                            labelStyle={'display': 'inline-block'},
                            inputStyle={"margin-left": "10px"},
                            
                        ),

                    ], xs=12, sm=12, md=12, lg=4, xl=4),
                    
                    
                    
                    
                    
                    dbc.Col([
                        dcc.Dropdown(
                            id='dd-souscripteurPolice',
                            options=[
                                {'label': i, 'value': i} for i in list_souscripteur
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
                        id = 'sousripteur-graph-id',
                        xs=12, sm=12, md=12, lg=12, xl=12  
                    ),
                ),
                html.Br(),
                dbc.Row(
                    dbc.Col(
                        #souscripteur_table(dt_ref,list_souscripteur[0]),
                        id = 'sousripteur-table-id',
                        xs=12, sm=12, md=12, lg=12, xl=12
                    )   
                )
            ]
            return  chld
        
        elif active_tab == "details-id":
            chld = [
                html.P(">>>>> Page Depréciée <<<<<"),
                # dbc.Row(
                #     dbc.Col(
                #         "Portefeuille détaillé par produits d'assurance en terme de nombre de police d'assurance émise, de montant de prime et de montant de prime encaissée.",
                #         className='h6',
                #         xs=12, sm=12, md=12, lg=12, xl=12
                #     )
                # ),
                # html.P(),
                # dbc.Row(
                #     dbc.Col([
                #         dcc.Dropdown(
                #             id='dd-prime_encaisse',
                #             options=[
                #                 {'label': i, 'value': i} for i in list_prime_encaisse
                #             ],
                #             value='nombre',
                #             clearable = False,
                #             style={
                #                 'font-size': "90%",
                #                 },
                #         )
                #     ], xs=12, sm=12, md=12, lg=4, xl=4),
                # ),
                # html.Br(),
                # dbc.Row(
                #     dbc.Col(
                #         #portfolio_details_graph2(ref,var, "branche", "prime"),
                #         id='branche-graph-id',
                #         xs=12, sm=12, md=12, lg=12, xl=12
                #     )

                # ),
                # html.Br(),
                # dbc.Row(
                #     dbc.Col(
                #         #portfolio_details_graph2(ref,var, "catégorie", "prime"),
                #         id='categorie-graph-id',
                #         xs=12, sm=12, md=12, lg=12, xl=12
                #     )
                # ),




                 
                 
                           
            ]



            return chld

        elif active_tab == "recouvrement-id":

            chld = [

                #dbc.Row(
                #    dbc.Col(
                #        "Saisie de recouvrement par client et par numero de police",
                #        className='h6',
                #        xs=12, sm=12, md=12, lg=12, xl=12
                #    )
                #),


                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Col([
                                    dcc.RadioItems(
                                        id = 'radio-recouvrement-s-maj-id',
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
                                    dbc.Button("RESET", id="button-reset", color="danger",  block=True, outline=True, size="sm"),
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
                                        html.H6('Id'),
                                        dcc.Input(
                                            id='input-recouvrement-id',
                                            placeholder='',
                                            type='number',
                                            style={'font-size': "90%",'color': '#504A4B'},
                                            className="m-1",
                                            #size='10',
                                            #value=funded_amount0
                                        ),  
                                    ],
                                    xs=12, sm=12, md=12, lg=6, xl=6
                                    ),

                                    dbc.Col([
                                            html.H6('Date de saisie'),
                                            dcc.Input(
                                                id='input-date-saisie',
                                                placeholder='date...',
                                                type='text',
                                                style={'font-size': "90%",'color': '#504A4B'},
                                                className="m-1",
                                                #size='10',
                                                #value=funded_amount0
                                            )  
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
                                html.H6('Souscripteur'),  
                                dcc.Dropdown(
                                    id = 'dd-souscripteur',
                                    options=[
                                            {'label': i, 'value': i} for i in list_souscripteur
                                    ],
                                    #value='souscripteur',
                                    optionHeight=45,
                                    clearable = False,
                                    style={
                                        'font-size': "90%",
                                        #'color': 'red',
                                    },
                                    className="m-1",  
                                ),
                            ], xs=12, sm=12, md=12, lg=3, xl=3),
                                
                            dbc.Col([
                                html.H6('Num. Police'),  
                                html.P(
                                    id = 'input-num-police',
                                    #placeholder='Num police',
                                    #value=X1_0,
                                    #style={'width': '100%', 'height': 150},
                                    style={'font-size': "90%",'color': 'grey'},
                                )


                            ], xs=12, sm=12, md=12, lg=3, xl=3),

                            dbc.Col([
                                html.H6('Prime impayée enregistrée'),  
                                dcc.Input(
                                    id='input-prime',
                                    placeholder='0',
                                    type='number',
                                    #value=funded_amount0
                                    style={'font-size': "90%",'color': 'grey'},
                                    className="m-1",
                                )  
                            ],xs=12, sm=12, md=12, lg=3, xl=3),

                            dbc.Col([
                                html.H6('Prime impayée actuelle'), 
                                html.H6(
                                    id='input-prime-solde',
                                    style={'font-size': "90%",'color': 'grey'}
                                ),
                                
                            ],xs=12, sm=12, md=12, lg=3, xl=3)              
                        ]),

                    ])
                ),
                
                html.Br(),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6('Garantie - Type'),  
                                dcc.Dropdown(
                                    id = 'dd-garantie-type',
                                    options=[
                                            {'label': i, 'value': i} for i in ['cheque','facture','traite_av','promesse']
                                    ],
                                    value='cheque',
                                    optionHeight=45,
                                    clearable = False,
                                    style={
                                        'font-size': "90%",
                                        'color': 'grey'
                                    },
                                    className="m-1",    
                                ),
                            ],xs=12, sm=12, md=12, lg=3, xl=3),

                            dbc.Col([
                                html.H6('Garantie - Montant'), 
                                dcc.Input(
                                    id='input-garantie-montant',
                                    placeholder='0',
                                    type='number',
                                    #value=funded_amount0,
                                    style={'font-size': "90%",'color': 'grey'},
                                    className="m-1", 
                                )  

                            ],xs=12, sm=12, md=12, lg=3, xl=3),

                            dbc.Col([
                                html.H6('Acompte - Montant'), 
                                dcc.Input(
                                    id='input-account-montant',
                                    placeholder='0',
                                    type='number',
                                    #value=funded_amount0,
                                    style={'font-size': "90%",'color': 'grey'},
                                    className="m-1",
                                )  

                            ],xs=12, sm=12, md=12, lg=3, xl=3),

                            dbc.Col([
                                html.H6('Acompte -  Référence'), 
                                dcc.Input(
                                    id='input-account-reference',
                                    placeholder='reçue CAR...',
                                    style={'font-size': "90%",'color': 'grey'},
                                    className="m-1",
                                    #type='number',
                                    #value=funded_amount0
                                )
                            ],xs=12, sm=12, md=12, lg=3, xl=3),
                        ]),

                    ])

                ),


                html.Br(),
                dbc.Row(
                    dbc.Col([
                         html.H6('Commentaires - Actions'),
                         dcc.Textarea(
                            id = 'input-commentaire',
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
                                html.H6('Situation du dossier'), 
                                dcc.Dropdown(
                                    id = 'dd-recouvrement-situation',
                                    options=[
                                            {'label': i, 'value': i} for i in ['Régularisé','Non-Régularisé']
                                    ],
                                    value='Non-Régularisé',
                                    #optionHeight=45,
                                    clearable = False,
                                    style={
                                        'font-size': "90%",
                                        'color': 'grey'
                                    }, 
                                    className="m-1",   
                                ),
                                html.Br()
                                

                            ])
                        )


                    ]),
                    
                    
                    
                    dbc.Col([    
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Col([
                                    html.Span(id = 'rappel-recouvrement',style={"color": "red"}),
                                    dbc.Button("VALIDATION", id="button-validation", color="primary",block=True),
                                    html.Span(id="example-output",style={'font-size': "90%"}),
                                 ]),
                            ])
                        ),
                    ]),


                    


                ]),
                



                
                
                html.Hr(),
                dbc.Row(
                    dbc.Col(
                        dbc.Button("Liste", id="button-list-id", color="primary",className="mr-2"),
                    ),
                ),
                dbc.Row(
                    dbc.Col(
                        #xdcc_table(df_recouvrement),
                        id="recouvrement-list-table-id",   
                    ),

                )
            
            ]
            return chld


        

            
    return ">>>> Sélectioner une rubrique"


# portefeuille
# @app.callback(
#     [
#        Output("card-souscripteur-id", "children"),
#        Output("card-police-id", "children"),
#        Output("card-quittance-id", "children"),
#        Output("card-p_souscrite-id", "children"),
#        Output("card-p_encaissee-id", "children"),
#        Output("card-p_impayee-id", "children"),
#        Output("new-quittance-id", "children"),

        
#     ],
#     [
#        Input("dd-dt-ref", "value"),
#        Input("dd-dt-var", "value"),
#        Input("dd-etat-quittance", "value"),

#     ]       
# )
# def card_update(d,dtv,eq): 
#     d = datetime.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
#     dtv = datetime.datetime.strptime(dtv, '%Y-%m-%dT%H:%M:%S')
#     res1 = card_content('souscripteur',d,dtv)
#     res2 = card_content('police',d,dtv)
#     res3 = card_content('quittance',d,dtv)
#     res4 = card_content('p_souscrite',d,dtv)
#     res5 = card_content('p_encaissee',d,dtv)
#     res6 = card_content('p_impayee',d,dtv)
#     #res7 = xdcc_table(new_quittance(d,dtv,eq)['df'][col_key])

#     res7 = [
#             dbc.Row([
#                         dbc.Col(
#                         html.P('Prime totale nette : {:,.0f} MGA (Ristourne: {:,.0f})'.format(new_quittance(d,dtv,eq)['sum_prime'], new_quittance(d,dtv,eq)['sum_ristourne'])),
#                         xs=12, sm=12, md=12, lg=4, xl=4
#                         ),
                        
#                         dbc.Col(
#                             html.P('Prime Payée: {:,.0f} MGA'.format(new_quittance(d,dtv,eq)['sum_payee'])),
#                             xs=12, sm=12, md=12, lg=4, xl=4
#                         ),
#                         dbc.Col(
#                             html.P('Prime Impayée: {:,.0f} MGA ({:,.0f}%)'.format(new_quittance(d,dtv,eq)['sum_impayee'],new_quittance(d,dtv,eq)['poids_impaye'])),
#                                 xs=12, sm=12, md=12, lg=4, xl=4
#                         ),
#                         #dbc.Col(html.P('Solde: {:,.0f} MGA'.format(renouvellement_table(dt,eq,m)['solde']))),

#                     ]),
            
#                     xdcc_table(new_quittance(d,dtv,eq)['df'][col_key])
#     ]

    
    
#     return res1,res2, res3,res4,res5, res6, res7

#souscripteur
@app.callback(  
    [   
        Output("sousripteur-graph-id", "children"),
        Output("sousripteur-table-id", "children"),
    ],  
    [
        Input("dd-dt-ref", "value"),
        Input("radio-souscripteurPolice-id","value"),
        Input("dd-souscripteurPolice", "value"),

    ]
)
def souscripteur_update(dt,s_p,v):
    if s_p == 'souscripteur':
        res1 = [
            dbc.Row([
                dbc.Col(
                    html.P('Prime totale nette : {:,.0f} MGA (Ristourne: {:,.0f})'.format(souscripteurPolice_graph (dt,s_p,v,None)['sum_prime'], souscripteurPolice_graph (dt,s_p,v,None)['sum_ristourne'],)),
                    xs=12, sm=12, md=12, lg=4, xl=4
                ),
                dbc.Col(
                    html.P('Prime Payée: {:,.0f} MGA'.format(souscripteurPolice_graph (dt,s_p,v,None)['sum_payee'])),
                    xs=12, sm=12, md=12, lg=4, xl=4
                ),
                dbc.Col(html.P('Prime Impayée: {:,.0f} MGA ({:,.0f}%)'.format(souscripteurPolice_graph (dt,s_p,v,None)['sum_impayee'],souscripteurPolice_graph (dt,s_p,v,None)['poids_impaye'])),
                    xs=12, sm=12, md=12, lg=4, xl=4
                ),
                #dbc.Col(html.P('Solde: {:,.0f} MGA'.format(souscripteurPolice_graph (dt,s_p,v,None)['solde'])),
                #    xs=12, sm=12, md=12, lg=3, xl=3
                #),

            ]),
            souscripteurPolice_graph (dt,s_p,v,None)['dcc_graph'],
        ]
        res2 = souscripteurPolice_table (dt,s_p,v,None)
    if s_p == 'police':
        res1 = souscripteurPolice_graph (dt,s_p,None,v)['dcc_graph']
        res2 = souscripteurPolice_table (dt,s_p,None,v)
    return res1, res2


#renouvelement
@app.callback(
    [
        Output("renouvellement-graph-id", "children"),
        Output("renouvellement-table-id", "children"),  
    ],
    [
        Input("dd-dt-ref", "value"),
        #Input("dd-etat", "value"), 
        Input("dd-etat-quittance", "value"),
        Input("dd-mois", "value"),

    ]
)
def renouvelement_update(dt,eq,m):
    res1 = renouvellement_graph(dt,'All','All','50%')
    res2 = [
            dbc.Row([
                        dbc.Col(
                        html.P('Prime totale nette : {:,.0f} MGA (Ristourne: {:,.0f})'.format(renouvellement_table (dt,eq,m)['sum_prime'], renouvellement_table (dt,eq,m)['sum_ristourne'])),
                        xs=12, sm=12, md=12, lg=4, xl=4
                        ),
                        
                        dbc.Col(
                            html.P('Prime Payée: {:,.0f} MGA'.format(renouvellement_table(dt,eq,m)['sum_payee'])),
                            xs=12, sm=12, md=12, lg=4, xl=4
                        ),
                        dbc.Col(
                            html.P('Prime Impayée: {:,.0f} MGA ({:,.0f}%)'.format(renouvellement_table(dt,eq,m)['sum_impayee'],renouvellement_table(dt,eq,m)['poids_impaye'])),
                                xs=12, sm=12, md=12, lg=4, xl=4
                        ),
                        #dbc.Col(html.P('Solde: {:,.0f} MGA'.format(renouvellement_table(dt,eq,m)['solde']))),

                    ]),
            
            xdcc_table(renouvellement_table(dt,eq,m)['df'])
    ]
    
    return res1, res2

#impayées
@app.callback(
    
        Output("impayees-graph-id", "children"),  
    [
        Input("dd-dt-ref", "value"),
    ]
)
def impayees_update(dt):
    res1 = impayees_graph(dt,'50%')
    return res1        


# #details
# @app.callback(
#     [
#         Output("branche-graph-id", "children"),
#         Output("categorie-graph-id", "children"),
#     ],
    
#         Input("dd-dt-ref", "value"),
#         Input("dd-dt-var", "value"),
#         Input("dd-prime_encaisse", "value"),

        
# )
# def details_update(ref,var,p):
#     res1 = [
#         dbc.Row([
#             dbc.Col(
#                 portfolio_details_graph3(ref,var, "branche", p,'level'),
#                 xs=12, sm=12, md=12, lg=6, xl=6
#             ),
#             dbc.Col(
#                 portfolio_details_graph3(ref,var, "branche", p,'var'),
#                 xs=12, sm=12, md=12, lg=6, xl=6
#             )
#         ])
#     ]
#     res2 = [
#         dbc.Row([
#             dbc.Col(
#                 portfolio_details_graph3(ref,var, "catégorie", p,'level'),
#                 xs=12, sm=12, md=12, lg=6, xl=6
#             ),
#             dbc.Col(
#                 portfolio_details_graph3(ref,var, "catégorie", p,'var'),
#                 xs=12, sm=12, md=12, lg=6, xl=6
#             )
#         ])


#     ]
#     return res1, res2        


## Souscripteur&Police
@app.callback(
    
        Output("dd-souscripteurPolice", "options"), 
        Input("radio-souscripteurPolice-id", "value"),

        
)
def details_update(s_p):
    if s_p == 'souscripteur':
        res1 = [{'label': i, 'value': i} for i in list_souscripteur]
    elif  s_p == 'police':
        res1 = [{'label': i, 'value': i} for i in list_num_police]
        

    return res1 


#recouvrement insert / update
@app.callback(
    Output("example-output", "children"), 
    [
     Input("button-validation", "n_clicks"),
     Input("radio-recouvrement-s-maj-id", "value"),
     Input("input-recouvrement-id", "value"),
     Input("input-date-saisie", "value"),
     Input("dd-souscripteur", "value"),
     Input("input-prime", "value"),
     Input("dd-garantie-type", "value"),
     Input("input-garantie-montant", "value"),
     Input("input-account-montant", "value"),
     Input("input-account-reference", "value"),
    Input("input-commentaire", "value"),
     Input("dd-recouvrement-situation", "value"),
    



    ]
)
def on_button_click(n,choice,id,dt,s,pri,gaT,gaM,acc,accRef,com,sit):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'button-validation' in changed_id:
        if choice == 'Nouvelle saisie':
            return recouvrement_insert(s,pri,gaT,gaM,acc,accRef,com,sit) 
        elif choice == 'Mise à jour':
            return recouvrement_update(id,dt,s,pri,gaT,gaM,acc,accRef,com,sit)  
    else:
        return "Verifier avant de validez svp !"

#recouvrement liste
@app.callback(
    Output("recouvrement-list-table-id", "children"), 
    
    Input("button-list-id", "n_clicks")

)
def on_button_click(n):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'button-list-id' in changed_id:
        return xdcc_table(recouvrement_list()) 

#recouvrement initialisation
@app.callback(
    [
        Output("input-recouvrement-id", "value"),
        Output("input-recouvrement-id",'disabled'),
        #Output("input-date-saisie",'value'),
        Output("input-date-saisie",'disabled'),

    ], 
    
    Input("radio-recouvrement-s-maj-id", "value")

)
def recouvrement_initiliation(v):
    if v == 'Nouvelle saisie':
        return 0, True, True
    else:
        return 1, False,False 



#recouvrement maj
@app.callback(
    [
        
        Output("input-date-saisie", "value"),
        Output("dd-souscripteur", "value"),
        Output("input-prime", "value"),
        Output("dd-garantie-type", "value"),
        Output("input-garantie-montant", "value"),
        Output("input-account-montant", "value"),
        Output("input-account-reference", "value"),
        Output("input-commentaire", "value"),
        Output("dd-recouvrement-situation", "value"),

    ], 
    
    Input("input-recouvrement-id", "value")

)
def recouvrement_maj(v):
    if v != 0:
       res = recouvrement_select(v)  
       res_date = res[0][1]
       res_sou  = res[0][2]
       res_pri  = res[0][3]
       res_gat  = res[0][4]
       res_gam  = res[0][5]
       res_acc  = res[0][6]
       res_accRef  = res[0][7]
       res_com  = res[0][8]
       res_sit  = res[0][9]
       return res_date,res_sou,res_pri,res_gat,res_gam, res_acc,res_accRef, res_com, res_sit
    else:
        res = recouvrement_reset_list.copy()
        res.pop(0)
        return res

#recouvrement reset
@app.callback(
       
        Output("radio-recouvrement-s-maj-id", "value"),
        Input("button-reset", "n_clicks"),

)
def recouvrement_reset(n):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'button-reset' in changed_id:
        return "Nouvelle saisie"
    else:
        return "Nouvelle saisie"


## recouvrement souscripteur > police
@app.callback(
        
        Output("input-num-police", "children"),    
       
        Input("dd-souscripteur", "value")
     
    
)
def recouvrement_dd_update(s):
    dfx = df1[df1['souscripteur']==s]
    s_p = dfx['Num police'].unique()
    s_p = list(s_p)
    res_pol = ', '.join(map(str, s_p))
    return res_pol



## recouvrement souscripteur + police => solde prime
@app.callback(
        
        Output("input-prime-solde", "children"),
    [    
        Input("dd-souscripteur", "value"),
        #Input("dd-num-police", "value")
    ]
)
def recouvrement_solde_prime(s):
    
    res = prime_nette(s)
    res = '{:,.0f}'.format(res)

    return res


## recouvrement Rappel avant validation
@app.callback(
        Output("rappel-recouvrement", "children"),  
        Input("radio-recouvrement-s-maj-id", "value")
)
def recouvrement_rappel(v):
    res = v
    return res


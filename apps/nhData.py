# standard package
import pandas as pd
import numpy as np

## function
def func_checking(agences):    
    data_checking = [
        [k,
        v[1],
        len(["containsNan" for x in v[2]['Code souscripteur'] if pd.isna(x)]),
        len(["contains:" for y in v[2]['Prime totale quittance'] if ":" in str(y) ]),
        ] for k,v in agences.items() if not v[2].empty
    ]
    
    checking = dict()
    df_checking = pd.DataFrame(data = data_checking , columns = ['code','agence','ErreurCS','ErreurPTQ'])
    checking = {
        'df':df_checking,
        'ErreurCS': df_checking['ErreurCS'].sum(),
        'ErreurPTQ': df_checking['ErreurPTQ'].sum(),
        'erreur' : df_checking['ErreurCS'].sum()+df_checking['ErreurCS'].sum()
    }
    
    
    return checking

def NumQuittanceSelectionForNan(df):
    lsNumQ = [row['Num quittance'] for index, row in df.iterrows() if pd.isna(row['Code souscripteur'])]
    return lsNumQ

def NumQuittanceselectionForDate(df):
    lsNumQ = [row['Num quittance'] for index, row in df.iterrows() if ":" in str(row['Prime totale quittance'])]
    return lsNumQ

def decalage (df,numQ, col):
    y = df[df['Num quittance'] == numQ]
    idx = y.index[0]
    colRef = y.columns.tolist().index(col)
    while colRef < len(y.columns)-1:
        df.loc[idx,y.columns[colRef]] = y[y.columns[colRef+1]].tolist()[0]
        colRef = colRef+1
    return df

def decalageMain(df,erreur):
    if erreur == "ErreurCS":
        ls = NumQuittanceSelectionForNan(df)
        for  l in ls:
            df = decalage (df,l, 'Code souscripteur')
    elif erreur == "ErreurPTQ":
        ls = NumQuittanceselectionForDate(df)
        for  l in ls:
            df = decalage (df,l, 'Adresse Ass')
    return df

def func_reguler(df_checking, agences):
    for i, r in df_checking.iterrows():
       # if not nh_agences[r['code']][2].empty():
            if r['ErreurCS']>0:
                agences[r['code']][2]= decalageMain(agences[r['code']][2],'ErreurCS')
            if r['ErreurPTQ']>0:
                agences[r['code']][2]= decalageMain(agences[r['code']][2],'ErreurPTQ')
    return agences

def change_type(agences):
#cols chmt de type "Prime totale quittance"  -> float
    for k,v in agences.items():
        v[2] = v[2].astype({"Prime totale quittance": float , "Code souscripteur": int ,"Montant Encaissé quittance": float})
    return agences
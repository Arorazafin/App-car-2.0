# standard package
import pandas as pd
import numpy as np

# date & time package
import datetime

## Conceptions

### Les Objets


class Assurance:
    
    def __init__(self, numPolice, dateEffetPolice, dateEcheancePolice, branche, categorie, client, quittances ):
        self.numPolice = numPolice
        self.dateEffetPolice = dateEffetPolice 
        self.dateEcheancePolice = dateEcheancePolice
        self.branche = branche
        self.categorie = categorie
        self.client = client
        self.quittances = quittances
        self.prime = None
        self.solde = None
        
        
        
    
  
        
    def calculSoldeAssurance(self,collQuittances):
        s = 0
        p = 0
        for v in self.quittances:
            s = s+collQuittances.collectionQuittances[v].solde
            p = p+collQuittances.collectionQuittances[v].prime
        
        self.prime = p
        self.solde = s
        
        
        return p, s
        

class Quittance:
    def __init__(self, numQuittance, dateEffetQuittance, dateEcheanceQuittance, assurances, typeMouvement,paiement,etatQuittance):
        self.numQuittance = numQuittance
        self.dateEffetQuittance = dateEffetQuittance
        self.dateEcheanceQuittance=dateEcheanceQuittance
        self.annee = dateEffetQuittance.year
        self.mois = datetime.datetime.strptime(str(dateEffetQuittance.year)+"."+str(dateEffetQuittance.month)+".1", "%Y.%m.%d")
        #self.mois = str(dateEffetQuittance.year)+"."+str(dateEffetQuittance.month)
        self.assurances = assurances
        self.typeMouvement = typeMouvement
        self.paiement = paiement
        self.etatQuittance=etatQuittance
        self.prime = self.calculSoldeQuittance()[0]
        self.solde = self.calculSoldeQuittance()[1]
       
        
    def calculSoldeQuittance(self):
        p = 0
        for i, v in self.typeMouvement.items():
            p = p + v.primeAPayer
        res1 = self.paiement.montant - p
        res0 = p
        return res0, res1
        

class TypeMouvement:
    def __init__(self, typeMvmt, numAvenant, primeAPayer):
        self.typeMvmt =typeMvmt
        self.numAvenant=numAvenant
        self.primeAPayer = primeAPayer
    

class Paiement:
    def __init__ (self, datePaiement, montant):
        self.datePaiement =datePaiement
        self.montant = montant
    

class Client:
    def __init__(self, numOrass, nom, adresse, tel, email, assurances):
        self.numOrass = numOrass
        self.nom = nom
        self.adresse = adresse
        self.tel =tel
        self.email = email
        self.assurances = assurances
        self.solde = None
        self.prime = None
        
        
    def calculSoldeClient(self, instCollContrats):
        s = 0
        p = 0
        for c in self.assurances:
            s = s + instCollContrats.collectionAssurances[c].solde
            p = p + instCollContrats.collectionAssurances[c].prime
        self.solde = s
        self.prime = p
        #s = sum( instCollContrats.collectionAssurances[c].solde for x in 
        return p, s

class Agence:
    def __init__ (self, df_agence):
        self.df_agence = df_agence                              
        self.code = self.df_agence.head(1)['Code intermédiaire'][0]
        self.nom =self.df_agence.head(1)['intermédiaire'][0]
        self.typeIntermediaire = self.df_agence.head(1)['Libellé type intermédiaire'][0]
        self.quittances=None
        self.assurances=None
        self.clients = None
        self.totalPrime = None
        self.totalSolde = None
        self.totalNombreQuittance = None
        self.totalNombreAssurance = None
        self.totalNombreClient = None
        self.verificationCoherence = None
        self.dateExtraction = None
        self.CodeNomAgence = None
        
        self.instQuittanceAgence = None
        
    def initAttribut(self):
        instQuittances = Quittances(self.df_agence)
        instQuittances.initCollection()
        instQuittances.calculStats()
        self.quittances = instQuittances
        
        instAssurances = Assurances(self.df_agence,instQuittances)
        instAssurances.initCollection()
        instAssurances.calculStats()
        self.assurances=instAssurances
        
        instClients = Clients(self.df_agence,instAssurances)
        instClients.initCollection()
        instClients.calculStats()
        self.clients=instClients
        
        
        ## verification des instances
        if (instQuittances.totalSolde - instAssurances.totalSolde == 0) & (instClients.totalSolde-instAssurances.totalSolde == 0) & (instQuittances.totalPrime - instAssurances.totalPrime == 0) & (instClients.totalPrime-instAssurances.totalPrime == 0):
            verif = "Cohérence OK"
        else:
            ls_c = [c.assurances for k,c in instClients.collectionClients.items()]
            ls_c = np.hstack(ls_c).tolist()
            seen=set()
            dup = [x for x in ls_c if x in seen or seen.add(x)]
            verif = "Données à revérifier: Doublon de contrat " + str(dup)
          
        
        self.verificationCoherence = verif
        
        #date d'extraction
        dateExtraction = self.df_agence['dateExtraction'].dt.date[0]
        agenceCode = str(self.df_agence.head(1)['Code intermédiaire'][0])
        agenceCode = agenceCode +'-'+ str(self.df_agence.head(1)['intermédiaire'][0])
        self.CodeNomAgence = agenceCode
        self.dateExtraction= dateExtraction
        
        #prime et solde
        self.totalPrime = instQuittances.totalPrime
        self.totalSolde = instQuittances.totalSolde
        
        #Nombre
        self.totalNombreQuittance = len(instQuittances.collectionQuittances)
        self.totalNombreAssurance = len(instAssurances.collectionAssurances)
        self.totalNombreClient = len(instClients.collectionClients)
        
        
        return self
    

    

#### Les Collections

class Agences:
    def __init__(self,agences):
        self.df_agences = agences
        self.collectionAgences = None
        self.totalNombreReseau = None
        
        self.tableauSynthese = dict() 
        
        self.totalPrime = dict()
        self.totalSolde = dict() 
    
        self.vueQuittancesNombre=dict()
        self.vueQuittancesEvolAnnuel = dict()
        self.vueQuittancesEvolAnnuelPct = dict()
        self.vueQuittancesEvolAnnuelTest = None
        
        #vue assurances
        self.vueAssurancesTotalNombre = dict()
        self.vueAssurancesTotalPrime = dict()
        self.vueAssurancesTotalSolde = dict()
        self.vueAssurancesDfBranche = dict()
        self.vueAssurancesBrancheRentable = dict()
        self.vueAssurancesDfCategorie = dict()
        self.vueAssurancesCategorieRentable = dict()
        
        #vue client
        self.vueClientsTotalNombre = dict()
        self.vueClientsTotalPrime = dict()
        self.vueClientsTotalSolde = dict()
        self.vueClientsDfClient = dict()
        self.vueClientsClientPlusRentable = dict()
        self.vueClientsClient2ePlusRentable = dict()
        self.vueClientsClientPlusGrosDeficit = dict()

    def vueQuittancesEvolAnnuel2(self,reseau,idx,critere):
        # idex = annee, mois ; criere = prime, solde 
        res = self.vueQuittancesEvolAnnuel[reseau].loc[idx,critere] 
        return res
        
    def initCollection(self):
        #df_agences
        data_agences = [ [k,v[2]] for k,v in self.df_agences.items()] 
        self.df_agences = pd.DataFrame(data=data_agences, columns = ['code','df_agence'] )
        df = pd.DataFrame(self.df_agences)
        df['objetAgence'] = df.apply(lambda x: Agence(x['df_agence']).initAttribut(),axis=1)
        df= df[['code','objetAgence']]
        
        #collectionAgences ET tableau de syntheses
        self.collectionAgences = df.set_index('code').to_dict()['objetAgence']
        df['type_intermediaire'] = df.apply(lambda x: x['objetAgence'].typeIntermediaire,axis =1)
        df['agence'] = df.apply(lambda x: x['objetAgence'].nom,axis =1)
        df['codeName'] = df.apply(lambda x: x['objetAgence'].CodeNomAgence,axis =1)
        df['prime'] = df.apply(lambda x: x['objetAgence'].totalPrime,axis =1)
        df['solde'] = df.apply(lambda x: x['objetAgence'].totalSolde,axis =1)
        df['tx_paiement'] = df.apply(lambda x: ((x['solde']+x['prime'])/x['prime']),axis =1)
        df['ranking'] = df.apply(lambda x: x['tx_paiement']*x['prime'],axis =1)
        # nombre reseau
        dfNb = df[['type_intermediaire','codeName']]
        dfNb = dfNb.groupby(['type_intermediaire','codeName'])['codeName'].count()
        dfNb = dfNb.reset_index(name="codeName2")
        self.totalNombreReseau = {
            'all':dfNb['codeName2'].sum(),
            'ac':dfNb[dfNb['type_intermediaire']=='Agence Centrale']['codeName2'].sum(),
            'ag':dfNb[dfNb['type_intermediaire']=='Agent Général']['codeName2'].sum(),

        }

        
        
        df1 = df[['type_intermediaire','agence','prime','solde','tx_paiement','ranking']]
        dfAll = df[['codeName','prime','solde','tx_paiement','ranking']]
        dfAc = df[df['type_intermediaire']=='Agence Centrale'][['codeName','prime','solde','tx_paiement','ranking']]
        dfAg = df[df['type_intermediaire']=='Agent Général'][['codeName','prime','solde','tx_paiement','ranking']]
        
        dfAll_data = [
            [
            'Réseau', 
            dfAll['prime'].sum(),
            dfAll['solde'].sum(),
            (dfAll['solde'].sum()+dfAll['prime'].sum())/dfAll['prime'].sum(),
            dfAll['prime'].sum()*((dfAll['solde'].sum()+dfAll['prime'].sum())/dfAll['prime'].sum())
            ],
            [
            'Agence Centrale', 
            dfAc['prime'].sum(),
            dfAc['solde'].sum(),
            (dfAc['solde'].sum()+dfAc['prime'].sum())/dfAc['prime'].sum(),
            dfAc['prime'].sum()*((dfAc['solde'].sum()+dfAc['prime'].sum())/dfAc['prime'].sum())
            ],
            [
            'Agence Générale', 
            dfAg['prime'].sum(),
            dfAg['solde'].sum(),
            (dfAg['solde'].sum()+dfAg['prime'].sum())/dfAg['prime'].sum(),
            dfAg['prime'].sum()*((dfAg['solde'].sum()+dfAg['prime'].sum())/dfAg['prime'].sum())
            ],

        ]

        dfAll = pd.DataFrame(data = dfAll_data, columns = dfAll.columns)
        self.tableauSynthese = {'all':dfAll}
        self.tableauSynthese.update ({'ac': dfAc})
        self.tableauSynthese.update ({'ag': dfAg})
        

            
        self.totalPrime.update({'all': df['prime'].sum()})
        #self.totalPrime.update ({'ac':df1})
        df1 = df1.groupby('type_intermediaire').sum()
        df1.reset_index(inplace = True)
        self.totalPrime.update ({'ac': df1[df1['type_intermediaire']=='Agence Centrale']['prime'].sum()})
        self.totalPrime.update ({'ag': df1[df1['type_intermediaire']=='Agent Général']['prime'].sum()})
        
        self.totalSolde.update({'all': df['solde'].sum()})
        self.totalSolde.update ({'ac': df1[df1['type_intermediaire']=='Agence Centrale']['solde'].sum()})
        self.totalSolde.update ({'ag': df1[df1['type_intermediaire']=='Agent Général']['solde'].sum()})
        
        
                
        #le DF de base
        x0 = pd.DataFrame.from_dict(self.collectionAgences, orient = 'index')
        x0.columns = ['agenceObjet']
        x0['type_intermediaire'] = x0.apply(lambda x: x['agenceObjet'].typeIntermediaire ,axis=1)  
        
        #VUE QUITTANCE
        x0Quit = x0.copy()
        x0Quit['quittancesEvolAnnuel'] = x0Quit.apply(lambda x: x['agenceObjet'].quittances.evolAnnuel.reset_index().values ,axis=1)
        x0Quit['quittancesNb'] = x0Quit.apply(lambda x: len(x['agenceObjet'].quittances.collectionQuittances) ,axis=1)
        
        #etape nombre
        x2 = x0Quit[['type_intermediaire','quittancesNb']]
        x2.reset_index(inplace = True)
        x2 = x2.drop('index',axis=1)
        x2 = x2.groupby('type_intermediaire').sum()
        x2.reset_index(inplace = True)
        self.vueQuittancesNombre.update({'all': x2['quittancesNb'].sum()})
        self.vueQuittancesNombre.update ({'ac': x2[x2['type_intermediaire']=='Agence Centrale']['quittancesNb'].sum()})
        self.vueQuittancesNombre.update ({'ag': x2[x2['type_intermediaire']=='Agent Général']['quittancesNb'].sum()})
        
        #etape quittancesEvolAnnuel
        x = x0Quit[['type_intermediaire','quittancesEvolAnnuel']]
        x = x.reset_index().set_index(['index','type_intermediaire'])
        x = x.explode('quittancesEvolAnnuel')
        x = x['quittancesEvolAnnuel'].apply(pd.Series).reset_index()
        x.columns = ['code','type_intermediaire','annee','prime', 'solde', 'paiement', 'tx_paiement']
        x = x [['type_intermediaire','annee','prime','solde', 'paiement', 'tx_paiement']]
        x = x.astype({'annee':int})
        x = x.groupby(['type_intermediaire','annee']).sum()
        x['tx_paiement']= x['paiement']/x['prime']
        #AC
        xAC = x.reset_index()
        xAC = xAC[xAC['type_intermediaire']=='Agence Centrale']
        xAC.drop('type_intermediaire', axis = 1, inplace = True)
        xAC.set_index('annee', inplace = True)
        xAC['tx_paiement']= xAC['paiement']/xAC['prime']
        self.vueQuittancesEvolAnnuel.update({'ac':xAC})
        
        #AG
        xAG = x.reset_index()
        xAG = xAG[xAG['type_intermediaire']=='Agent Général']
        xAG.drop('type_intermediaire', axis = 1, inplace = True)
        xAG.set_index('annee', inplace = True)
        xAG['tx_paiement']= xAG['paiement']/xAG['prime']
        self.vueQuittancesEvolAnnuel.update({'ag':xAG})
                
        #toute Agence
        xAll = x.reset_index()
        xAll.drop('type_intermediaire', axis =1, inplace = True)
        xAll = xAll.groupby(['annee']).sum()
        xAll['tx_paiement']= xAll['paiement']/xAll['prime']
        self.vueQuittancesEvolAnnuel.update({'all':xAll})
 
        #pourcentage
        xAcBis =  xAC/xAll
        xAcBis = xAcBis.fillna(0)
        xAcBis['tx_paiement'] = None
        self.vueQuittancesEvolAnnuelPct.update({'ac':xAcBis})
        xAgBis  =  xAG/xAll
        xAgBis = xAgBis.fillna(0)
        xAgBis['tx_paiement'] = None
        self.vueQuittancesEvolAnnuelPct.update({'ag':xAgBis})
        
        xAllBis = xAll/xAll
        xAllBis = xAllBis.fillna(0)
        xAllBis['tx_paiement'] = None
        self.vueQuittancesEvolAnnuelPct.update({'all':xAllBis})
        
        #test
        vueQuittancesEvolAnnuelTest = self.totalPrime['all'] == self.vueQuittancesEvolAnnuel['all']['prime'].sum()
        self.vueQuittancesEvolAnnuelTest = vueQuittancesEvolAnnuelTest
        
        



        #VUE ASSURANCE
        x0Ass = x0.copy()
        x0Ass['assurancesStats'] = x0Ass.apply(lambda x: x['agenceObjet'].assurances.stats.reset_index().values ,axis=1)
        x0Ass['assurancesNb'] = x0Ass.apply(lambda x: len(x['agenceObjet'].assurances.collectionAssurances) ,axis=1)

        #etape nombre
        vueAssurancesNombre = dict()
        x2Ass = x0Ass[['type_intermediaire','assurancesNb']]
        x2Ass.reset_index(inplace = True)
        x2Ass = x2Ass.drop('index',axis=1)
        x2Ass = x2Ass.groupby('type_intermediaire').sum()
        x2Ass.reset_index(inplace = True)
        vueAssurancesNombre.update({'all': x2Ass['assurancesNb'].sum()})
        vueAssurancesNombre.update ({'ac': x2Ass[x2Ass['type_intermediaire']=='Agence Centrale']['assurancesNb'].sum()})
        vueAssurancesNombre.update ({'ag': x2Ass[x2Ass['type_intermediaire']=='Agent Général']['assurancesNb'].sum()})
        self.vueAssurancesTotalNombre = vueAssurancesNombre

        #etape stats
        x = x0Ass[['type_intermediaire','assurancesStats']]
        x = x.reset_index().set_index(['index','type_intermediaire'])
        x = x.explode('assurancesStats')
        x = x['assurancesStats'].apply(pd.Series).reset_index()
        x.columns = ['code','type_intermediaire','idx','numPolice', 'branche', 'categorie','prime','solde']
        x.drop('idx', axis =1, inplace = True)

        #total prime
        vueAssurancesTotalPrime = dict()
        x2Ass = x[['type_intermediaire','prime']]
        x2Ass.reset_index(inplace = True)
        x2Ass = x2Ass.drop('index',axis=1)
        x2Ass = x2Ass.groupby('type_intermediaire').sum()
        x2Ass.reset_index(inplace = True)
        vueAssurancesTotalPrime.update({'all': x2Ass['prime'].sum()})
        vueAssurancesTotalPrime.update ({'ac': x2Ass[x2Ass['type_intermediaire']=='Agence Centrale']['prime'].sum()})
        vueAssurancesTotalPrime.update ({'ag': x2Ass[x2Ass['type_intermediaire']=='Agent Général']['prime'].sum()})
        self.vueAssurancesTotalPrime = vueAssurancesTotalPrime

        #total solde
        vueAssurancesTotalSolde = dict()
        x2Ass = x[['type_intermediaire','solde']]
        x2Ass.reset_index(inplace = True)
        x2Ass = x2Ass.drop('index',axis=1)
        x2Ass = x2Ass.groupby('type_intermediaire').sum()
        x2Ass.reset_index(inplace = True)
        vueAssurancesTotalSolde.update({'all': x2Ass['solde'].sum()})
        vueAssurancesTotalSolde.update ({'ac': x2Ass[x2Ass['type_intermediaire']=='Agence Centrale']['solde'].sum()})
        vueAssurancesTotalSolde.update ({'ag': x2Ass[x2Ass['type_intermediaire']=='Agent Général']['solde'].sum()})
        self.vueAssurancesTotalSolde = vueAssurancesTotalSolde



        #Branche
        brancheNbDf = pd.DataFrame(x[['type_intermediaire','branche']].value_counts(normalize=False))
        branchePrime = pd.DataFrame(x.groupby(['type_intermediaire','branche'])['prime'].sum()).sort_values(by='prime',ascending = False)
        brancheSolde = pd.DataFrame(x.groupby(['type_intermediaire','branche'])['solde'].sum()).sort_values(by='solde',ascending = True)
        df_branche = brancheNbDf.join(branchePrime)
        df_branche = df_branche.join(brancheSolde)
        df_branche ['tx_paiement'] = (df_branche['prime']+df_branche['solde'])/df_branche['prime']
        df_branche ['ranking']= df_branche.apply(lambda x: (x['prime']*x['tx_paiement'] if x['solde']<=0 else 0), axis =1)
        df_branche.columns = ['nombre','prime','solde','tx_paiement','ranking']
        df_branche.reset_index(inplace = True)

        #All
        df_brancheAll = df_branche [['branche','nombre','prime','solde','tx_paiement','ranking']]
        df_brancheAll.set_index('branche', inplace = True)
        dfBranche = {'all':df_brancheAll}

        #branche AC
        df_brancheAC = df_branche[df_branche['type_intermediaire']=='Agence Centrale']
        df_brancheAC = df_brancheAC [['branche','nombre','prime','solde','tx_paiement','ranking']]
        df_brancheAC.set_index('branche', inplace = True)
        dfBranche.update({'ac':df_brancheAC})

        #branche AG
        df_brancheAG = df_branche[df_branche['type_intermediaire']=='Agent Général']
        df_brancheAG = df_brancheAG [['branche','nombre','prime','solde','tx_paiement','ranking']]
        df_brancheAG.set_index('branche', inplace = True)
        dfBranche.update({'ag':df_brancheAG})
        self.vueAssurancesDfBranche = dfBranche

        # meilleur branche
        brancheRentable = dict()
        ## meilleur branche all
        dfx1 = df_brancheAll.copy()
        dfx1.reset_index(inplace = True)
        dfx1 = dfx1.sort_values(by ='ranking', ascending = False)
        brancheBestProduct = dfx1.head(1)['branche'].tolist()[0]
        brancheBestNb = dfx1.head(1)['nombre'].tolist()[0]
        brancheBestNbPercentage = (dfx1.head(1)['nombre'] / dfx1['nombre'].sum()).tolist()[0]
        brancheBestPrime = dfx1.head(1)['prime'].tolist()[0]
        brancheBestPrimePercentage = (dfx1.head(1)['prime'] / dfx1['prime'].sum()).tolist()[0]
        brancheBestTxPaiement = dfx1.head(1)['tx_paiement'].tolist()[0]
        brancheRentable.update( 
            {'all': 
             {'product':brancheBestProduct,
                                        'nb' : brancheBestNb,
                                        'nbPercentage' : brancheBestNbPercentage,
                                        'prime' : brancheBestPrime,
                                        'primePercentage':brancheBestPrimePercentage,
                                        'txPaiement': brancheBestTxPaiement
             }
            }
        )


        ## meilleur branche ac
        dfx1 = df_brancheAC.copy()
        dfx1.reset_index(inplace = True)
        dfx1 = dfx1.sort_values(by ='ranking', ascending = False)
        brancheBestProduct = dfx1.head(1)['branche'].tolist()[0]
        brancheBestNb = dfx1.head(1)['nombre'].tolist()[0]
        brancheBestNbPercentage = (dfx1.head(1)['nombre'] / dfx1['nombre'].sum()).tolist()[0]
        brancheBestPrime = dfx1.head(1)['prime'].tolist()[0]
        brancheBestPrimePercentage = (dfx1.head(1)['prime'] / dfx1['prime'].sum()).tolist()[0]
        brancheBestTxPaiement = dfx1.head(1)['tx_paiement'].tolist()[0]
        brancheRentable.update( 
            {'ac': 
             {'product':brancheBestProduct,
                                        'nb' : brancheBestNb,
                                        'nbPercentage' : brancheBestNbPercentage,
                                        'prime' : brancheBestPrime,
                                        'primePercentage':brancheBestPrimePercentage,
                                        'txPaiement': brancheBestTxPaiement
             }
            }
        )



        ## meilleur branche ag
        dfx1 = df_brancheAG.copy()
        dfx1.reset_index(inplace = True)
        dfx1 = dfx1.sort_values(by ='ranking', ascending = False)
        try:
            brancheBestProduct = dfx1.head(1)['branche'].tolist()[0]
            brancheBestNb = dfx1.head(1)['nombre'].tolist()[0]
            brancheBestNbPercentage = (dfx1.head(1)['nombre'] / dfx1['nombre'].sum()).tolist()[0]
            brancheBestPrime = dfx1.head(1)['prime'].tolist()[0]
            brancheBestPrimePercentage = (dfx1.head(1)['prime'] / dfx1['prime'].sum()).tolist()[0]
            brancheBestTxPaiement = dfx1.head(1)['tx_paiement'].tolist()[0]
            brancheRentable.update( 
                {'ag': 
                 {'product':brancheBestProduct,
                                        'nb' : brancheBestNb,
                                        'nbPercentage' : brancheBestNbPercentage,
                                        'prime' : brancheBestPrime,
                                        'primePercentage':brancheBestPrimePercentage,
                                        'txPaiement': brancheBestTxPaiement
                 }
                }
            )
        except:
            brancheRentable.update( 
                {'ag': 
                 {'product':None,
                                        'nb' : None,
                                        'nbPercentage' : None,
                                        'prime' : None,
                                        'primePercentage':None,
                                        'txPaiement': None
                 }
                }
            )
        self.vueAssurancesBrancheRentable = brancheRentable
            
            
            
        #Categorie
        categorieNbDf = pd.DataFrame(x[['type_intermediaire','categorie']].value_counts(normalize=False))
        categoriePrime = pd.DataFrame(x.groupby(['type_intermediaire','categorie'])['prime'].sum()).sort_values(by='prime',ascending = False)
        categorieSolde = pd.DataFrame(x.groupby(['type_intermediaire','categorie'])['solde'].sum()).sort_values(by='solde',ascending = True)
        df_categorie = categorieNbDf.join(categoriePrime)
        df_categorie = df_categorie.join(categorieSolde)
        df_categorie ['tx_paiement'] = (df_categorie['prime']+df_categorie['solde'])/df_categorie['prime']
        df_categorie ['ranking']= df_categorie.apply(lambda x: (x['prime']*x['tx_paiement'] if x['solde']<=0 else 0), axis =1)
        df_categorie.columns = ['nombre','prime','solde','tx_paiement','ranking']
        df_categorie.reset_index(inplace = True)

        #All
        df_categorieAll = df_categorie [['categorie','nombre','prime','solde','tx_paiement','ranking']]
        df_categorieAll.set_index('categorie', inplace = True)
        dfCategorie = {'all':df_categorieAll}

        #AC
        df_categorieAC = df_categorie[df_categorie['type_intermediaire']=='Agence Centrale']
        df_categorieAC = df_categorieAC [['categorie','nombre','prime','solde','tx_paiement','ranking']]
        df_categorieAC.set_index('categorie', inplace = True)
        dfCategorie.update({'ac':df_categorieAC})

        #AG
        df_categorieAG = df_categorie[df_categorie['type_intermediaire']=='Agent Général']
        df_categorieAG = df_categorieAG [['categorie','nombre','prime','solde','tx_paiement','ranking']]
        df_categorieAG.set_index('categorie', inplace = True)
        dfCategorie.update({'ag':df_categorieAG})
        
        self.vueAssurancesDfCategorie = dfCategorie
        

        # meilleur categorie
        categorieRentable = dict()
        ## meilleur categorie all
        dfx1 = df_categorieAll.copy()
        dfx1.reset_index(inplace = True)
        dfx1 = dfx1.sort_values(by ='ranking', ascending = False)
        categorieBestProduct = dfx1.head(1)['categorie'].tolist()[0]
        categorieBestNb = dfx1.head(1)['nombre'].tolist()[0]
        categorieBestNbPercentage = (dfx1.head(1)['nombre'] / dfx1['nombre'].sum()).tolist()[0]
        categorieBestPrime = dfx1.head(1)['prime'].tolist()[0]
        categorieBestPrimePercentage = (dfx1.head(1)['prime'] / dfx1['prime'].sum()).tolist()[0]
        categorieBestTxPaiement = dfx1.head(1)['tx_paiement'].tolist()[0]
        categorieRentable.update( 
            {'all': 
             {'product':categorieBestProduct,
                                        'nb' : categorieBestNb,
                                        'nbPercentage' : categorieBestNbPercentage,
                                        'prime' : categorieBestPrime,
                                        'primePercentage':categorieBestPrimePercentage,
                                        'txPaiement': categorieBestTxPaiement
             }
            }
        )


        ## meilleur categorie ac
        dfx1 = df_categorieAC.copy()
        dfx1.reset_index(inplace = True)
        dfx1 = dfx1.sort_values(by ='ranking', ascending = False)
        categorieBestProduct = dfx1.head(1)['categorie'].tolist()[0]
        categorieBestNb = dfx1.head(1)['nombre'].tolist()[0]
        categorieBestNbPercentage = (dfx1.head(1)['nombre'] / dfx1['nombre'].sum()).tolist()[0]
        categorieBestPrime = dfx1.head(1)['prime'].tolist()[0]
        categorieBestPrimePercentage = (dfx1.head(1)['prime'] / dfx1['prime'].sum()).tolist()[0]
        categorieBestTxPaiement = dfx1.head(1)['tx_paiement'].tolist()[0]
        categorieRentable.update( 
            {'ac': 
             {'product':categorieBestProduct,
                                        'nb' : categorieBestNb,
                                        'nbPercentage' : categorieBestNbPercentage,
                                        'prime' : categorieBestPrime,
                                        'primePercentage':categorieBestPrimePercentage,
                                        'txPaiement': categorieBestTxPaiement
             }
            }
        )




        ## meilleur categorie ag
        dfx1 = df_categorieAG.copy()
        dfx1.reset_index(inplace = True)
        dfx1 = dfx1.sort_values(by ='ranking', ascending = False)
        try:    
            categorieBestProduct = dfx1.head(1)['categorie'].tolist()[0]
            categorieBestNb = dfx1.head(1)['nombre'].tolist()[0]
            categorieBestNbPercentage = (dfx1.head(1)['nombre'] / dfx1['nombre'].sum()).tolist()[0]
            categorieBestPrime = dfx1.head(1)['prime'].tolist()[0]
            categorieBestPrimePercentage = (dfx1.head(1)['prime'] / dfx1['prime'].sum()).tolist()[0]
            categorieBestTxPaiement = dfx1.head(1)['tx_paiement'].tolist()[0]
            categorieRentable.update( 
                {'ag': 
                 {'product':categorieBestProduct,
                                            'nb' : categorieBestNb,
                                            'nbPercentage' : categorieBestNbPercentage,
                                            'prime' : categorieBestPrime,
                                            'primePercentage':categorieBestPrimePercentage,
                                            'txPaiement': categorieBestTxPaiement
                 }
                }
            )

        except:
            categorieRentable.update( 
                {'ag': 
                 {'product':None,
                                        'nb' : None,
                                        'nbPercentage' : None,
                                        'prime' : None,
                                        'primePercentage':None,
                                        'txPaiement': None
                 }
                }
            )


        self.vueAssurancesCategorieRentable = categorieRentable
        
        
        ## VUE CLIENT
        x0Client = x0.copy()
        x0Client['clientStats'] = x0Client.apply(lambda x: x['agenceObjet'].clients.dfClient.reset_index().values ,axis=1)
        x0Client['clientNb'] = x0Client.apply(lambda x: len(x['agenceObjet'].clients.collectionClients) ,axis=1)
                                         

        #etape nombre
        vueClientsTotalNombre = dict()
        x2 = x0Client[['type_intermediaire','clientNb']]
        x2.reset_index(inplace = True)
        x2 = x2.drop('index',axis=1)
        x2 = x2.groupby('type_intermediaire').sum()
        x2.reset_index(inplace = True)
        vueClientsTotalNombre.update({'all': x2['clientNb'].sum()})
        vueClientsTotalNombre.update ({'ac': x2[x2['type_intermediaire']=='Agence Centrale']['clientNb'].sum()})
        vueClientsTotalNombre.update ({'ag': x2[x2['type_intermediaire']=='Agent Général']['clientNb'].sum()})
        self.vueClientsTotalNombre = vueClientsTotalNombre

        #etape stats
        x = x0Client[['type_intermediaire','clientStats']]
        x = x.reset_index().set_index(['index','type_intermediaire'])
        x = x.explode('clientStats')
        x = x['clientStats'].apply(pd.Series).reset_index()
        x.columns = ['code','type_intermediaire','idx','numClient', 'nom', 'prime','solde','5','6']
        x.drop(['idx','5','6'], axis =1, inplace = True)

        #total prime
        vueClientsTotalPrime = dict()
        x3 = x[['type_intermediaire','prime']]
        x3.reset_index(inplace = True)
        x3 = x3.drop('index',axis=1)
        x3 = x3.groupby('type_intermediaire').sum()
        x3.reset_index(inplace = True)
        vueClientsTotalPrime.update({'all': x3['prime'].sum()})
        vueClientsTotalPrime.update ({'ac': x3[x3['type_intermediaire']=='Agence Centrale']['prime'].sum()})
        vueClientsTotalPrime.update ({'ag': x3[x3['type_intermediaire']=='Agent Général']['prime'].sum()})
        self.vueClientsTotalPrime = vueClientsTotalPrime

        #total prime
        vueClientsTotalSolde = dict()
        x3 = x[['type_intermediaire','solde']]
        x3.reset_index(inplace = True)
        x3 = x3.drop('index',axis=1)
        x3 = x3.groupby('type_intermediaire').sum()
        x3.reset_index(inplace = True)
        vueClientsTotalSolde.update({'all': x3['solde'].sum()})
        vueClientsTotalSolde.update ({'ac': x3[x3['type_intermediaire']=='Agence Centrale']['solde'].sum()})
        vueClientsTotalSolde.update ({'ag': x3[x3['type_intermediaire']=='Agent Général']['solde'].sum()})
        self.vueClientsTotalSolde = vueClientsTotalSolde


        vueClientsDfClient = dict()
        x['tx_paiement'] = (x['prime']+x['solde'])/x['prime']
        x ['ranking']= x.apply(lambda x: (x['prime']*x['tx_paiement'] if x['solde']<=0 else 0), axis =1)
        #all
        dfClient = x.copy()
        dfClient = dfClient[['code','numClient', 'nom', 'prime', 'solde','tx_paiement', 'ranking']]
        vueClientsDfClient.update({'all': dfClient})
        #ac
        dfClientAC = x.copy()
        dfClientAC = dfClientAC[dfClientAC['type_intermediaire']=='Agence Centrale']
        dfClientAC = dfClientAC[['code','numClient', 'nom', 'prime', 'solde','tx_paiement', 'ranking']]
        vueClientsDfClient.update({'ac': dfClientAC})
        #ag
        dfClientAG = x.copy()
        dfClientAG = dfClientAG[dfClientAG['type_intermediaire']=='Agent Général']
        dfClientAG = dfClientAG[['code','numClient', 'nom', 'prime', 'solde','tx_paiement', 'ranking']]
        vueClientsDfClient.update({'ag': dfClientAG})
        self.vueClientsDfClient = vueClientsDfClient

        #client le plus rentable
        vueClientsClientPlusRentable = dict()
        #ALL
        dfx = dfClient.sort_values(by ='ranking', ascending = False)
        clientPlusRentableNom  = dfx.head(1)['nom'].tolist()[0]
        clientPlusRentablePrime = dfx.head(1)['prime'].tolist()[0]
        clientPlusRentablePrimePercentage = (dfx.head(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        clientPlusRentableTxPaiement = dfx.head(1)['tx_paiement'].tolist()[0]
        clientPlusRentableSolde = dfx.head(1)['solde'].tolist()[0]
        clientPlusRentableCodeAgence = dfx.head(1)['code'].tolist()[0]

        vueClientsClientPlusRentable.update(
            {'all':
             {'nom':clientPlusRentableNom,
              'prime' : clientPlusRentablePrime,
              'primePercentage':clientPlusRentablePrimePercentage,
              'txPaiement': clientPlusRentableTxPaiement,
              'solde': clientPlusRentableSolde,
              'codeAgence': clientPlusRentableCodeAgence
             }
            }
        )

        #AC
        dfx = dfClientAC.sort_values(by ='ranking', ascending = False)
        clientPlusRentableNom  = dfx.head(1)['nom'].tolist()[0]
        clientPlusRentablePrime = dfx.head(1)['prime'].tolist()[0]
        clientPlusRentablePrimePercentage = (dfx.head(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        clientPlusRentableTxPaiement = dfx.head(1)['tx_paiement'].tolist()[0]
        clientPlusRentableSolde = dfx.head(1)['solde'].tolist()[0]
        clientPlusRentableCodeAgence = dfx.head(1)['code'].tolist()[0]

        vueClientsClientPlusRentable.update(
            {'ac':
             {'nom':clientPlusRentableNom,
              'prime' : clientPlusRentablePrime,
              'primePercentage':clientPlusRentablePrimePercentage,
              'txPaiement': clientPlusRentableTxPaiement,
              'solde': clientPlusRentableSolde,
              'codeAgence': clientPlusRentableCodeAgence
             }
            }
        )

        #AG
        dfx = dfClientAG.sort_values(by ='ranking', ascending = False)
        clientPlusRentableNom  = dfx.head(1)['nom'].tolist()[0]
        clientPlusRentablePrime = dfx.head(1)['prime'].tolist()[0]
        clientPlusRentablePrimePercentage = (dfx.head(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        clientPlusRentableTxPaiement = dfx.head(1)['tx_paiement'].tolist()[0]
        clientPlusRentableSolde = dfx.head(1)['solde'].tolist()[0]
        clientPlusRentableCodeAgence = dfx.head(1)['code'].tolist()[0]

        vueClientsClientPlusRentable.update(
            {'ag':
             {'nom':clientPlusRentableNom,
              'prime' : clientPlusRentablePrime,
              'primePercentage':clientPlusRentablePrimePercentage,
              'txPaiement': clientPlusRentableTxPaiement,
              'solde': clientPlusRentableSolde,
              'codeAgence': clientPlusRentableCodeAgence
             }
            }
        )

        self.vueClientsClientPlusRentable = vueClientsClientPlusRentable


        #client 2e plus rentable
        vueClientsClient2ePlusRentable = dict()
        #ALL
        dfx = dfClient.sort_values(by ='ranking', ascending = False)
        client2ePlusRentableNom  =dfx.head(2).tail(1)['nom'].tolist()[0]
        client2ePlusRentablePrime = dfx.head(2).tail(1)['prime'].tolist()[0]
        client2ePlusRentablePrimePercentage = (dfx.head(2).tail(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        client2ePlusRentableTxPaiement = dfx.head(2).tail(1)['tx_paiement'].tolist()[0]
        client2ePlusRentableSolde = dfx.head(2).tail(1)['solde'].tolist()[0]
        client2ePlusRentableCodeAgence = dfx.head(2).tail(1)['code'].tolist()[0]

        vueClientsClient2ePlusRentable.update(
            {'all':
             {'nom':client2ePlusRentableNom,
              'prime' : client2ePlusRentablePrime,
              'primePercentage':client2ePlusRentablePrimePercentage,
              'txPaiement': client2ePlusRentableTxPaiement,
              'solde': client2ePlusRentableSolde,
              'codeAgence': client2ePlusRentableCodeAgence
             }
            }
        )

        #AC
        dfx = dfClientAC.sort_values(by ='ranking', ascending = False)
        clientPlusRentableNom  = dfx.head(2).tail(1)['nom'].tolist()[0]
        clientPlusRentablePrime = dfx.head(2).tail(1)['prime'].tolist()[0]
        clientPlusRentablePrimePercentage = (dfx.head(2).tail(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        clientPlusRentableTxPaiement = dfx.head(2).tail(1)['tx_paiement'].tolist()[0]
        clientPlusRentableSolde = dfx.head(2).tail(1)['solde'].tolist()[0]
        clientPlusRentableCodeAgence = dfx.head(2).tail(1)['code'].tolist()[0]

        vueClientsClient2ePlusRentable.update(
            {'ac':
             {'nom':client2ePlusRentableNom,
              'prime' : client2ePlusRentablePrime,
              'primePercentage':client2ePlusRentablePrimePercentage,
              'txPaiement': client2ePlusRentableTxPaiement,
              'solde': client2ePlusRentableSolde,
              'codeAgence': client2ePlusRentableCodeAgence
             }
            }
        )

        #AG
        dfx = dfClientAG.sort_values(by ='ranking', ascending = False)
        client2ePlusRentableNom  = dfx.head(2).tail(1)['nom'].tolist()[0]
        client2ePlusRentablePrime = dfx.head(2).tail(1)['prime'].tolist()[0]
        client2ePlusRentablePrimePercentage = (dfx.head(2).tail(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        client2ePlusRentableTxPaiement = dfx.head(2).tail(1)['tx_paiement'].tolist()[0]
        client2ePlusRentableSolde = dfx.head(2).tail(1)['solde'].tolist()[0]
        client2ePlusRentableCodeAgence = dfx.head(2).tail(1)['code'].tolist()[0]

        vueClientsClient2ePlusRentable.update(
            {'ag':
             {'nom':client2ePlusRentableNom,
              'prime' : client2ePlusRentablePrime,
              'primePercentage':client2ePlusRentablePrimePercentage,
              'txPaiement': client2ePlusRentableTxPaiement,
              'solde': client2ePlusRentableSolde,
              'codeAgence': client2ePlusRentableCodeAgence
             }
            }
        )

        self.vueClientsClient2ePlusRentable=vueClientsClient2ePlusRentable



        #client le plus gros deficit
        vueClientsClientPlusGrosDeficit = dict()
        #ALL
        dfx = dfClient.sort_values(by ='solde', ascending = True)
        ClientPlusGrosDeficitNom  = dfx.head(1)['nom'].tolist()[0]
        ClientPlusGrosDeficitPrime = dfx.head(1)['prime'].tolist()[0]
        ClientPlusGrosDeficitPrimePercentage = (dfx.head(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        ClientPlusGrosDeficitTxPaiement = dfx.head(1)['tx_paiement'].tolist()[0]
        ClientPlusGrosDeficitSolde = dfx.head(1)['solde'].tolist()[0]
        ClientPlusGrosDeficitCodeAgence = dfx.head(1)['code'].tolist()[0]

        vueClientsClientPlusGrosDeficit.update(
            {'all':
             {'nom':ClientPlusGrosDeficitNom,
              'prime' : ClientPlusGrosDeficitPrime,
              'primePercentage':ClientPlusGrosDeficitPrimePercentage,
              'txPaiement': ClientPlusGrosDeficitTxPaiement,
              'solde': ClientPlusGrosDeficitSolde,
              'codeAgence': ClientPlusGrosDeficitCodeAgence
             }
            }
        )

        #AC
        dfx = dfClientAC.sort_values(by ='solde', ascending = True)
        ClientPlusGrosDeficitNom  = dfx.head(1)['nom'].tolist()[0]
        ClientPlusGrosDeficitPrime = dfx.head(1)['prime'].tolist()[0]
        ClientPlusGrosDeficitPrimePercentage = (dfx.head(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        ClientPlusGrosDeficitTxPaiement = dfx.head(1)['tx_paiement'].tolist()[0]
        ClientPlusGrosDeficitSolde = dfx.head(1)['solde'].tolist()[0]
        ClientPlusGrosDeficitCodeAgence = dfx.head(1)['code'].tolist()[0]

        vueClientsClientPlusGrosDeficit.update(
            {'ac':
             {'nom':ClientPlusGrosDeficitNom,
              'prime' : ClientPlusGrosDeficitPrime,
              'primePercentage':ClientPlusGrosDeficitPrimePercentage,
              'txPaiement': ClientPlusGrosDeficitTxPaiement,
              'solde': ClientPlusGrosDeficitSolde,
              'codeAgence': ClientPlusGrosDeficitCodeAgence
             }
            }
        )

        #AG
        dfx = dfClientAG.sort_values(by ='solde', ascending = True)
        ClientPlusGrosDeficitNom  = dfx.head(1)['nom'].tolist()[0]
        ClientPlusGrosDeficitPrime = dfx.head(1)['prime'].tolist()[0]
        ClientPlusGrosDeficitPrimePercentage = (dfx.head(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        ClientPlusGrosDeficitTxPaiement = dfx.head(1)['tx_paiement'].tolist()[0]
        ClientPlusGrosDeficitSolde = dfx.head(1)['solde'].tolist()[0]
        ClientPlusGrosDeficitCodeAgence = dfx.head(1)['code'].tolist()[0]

        vueClientsClientPlusGrosDeficit.update(
            {'ag':
             {'nom':ClientPlusGrosDeficitNom,
              'prime' : ClientPlusGrosDeficitPrime,
              'primePercentage':ClientPlusGrosDeficitPrimePercentage,
              'txPaiement': ClientPlusGrosDeficitTxPaiement,
              'solde': ClientPlusGrosDeficitSolde,
              'codeAgence': ClientPlusGrosDeficitCodeAgence
             }
            }
        )
        self.vueClientsClientPlusGrosDeficit =vueClientsClientPlusGrosDeficit





        
        return
    
    
 

class Quittances:
    def __init__ (self, df):
        self.df = df
        self.collectionQuittances = None
        self.totalSolde = None
        self.totalPrime = None
        self.totalNombre= None
        self.evolAnnuel = None
        self.evolMensuel = None
    
    def initCollection(self):
        #supprimer les doublons num quittance
        dfy = self.df.drop_duplicates(subset=['Num quittance'])
        dfy = dfy[['Num quittance','Date effet quittance','Date échéance quittance','Num police','Date Encaiss quittance','Montant Encaissé quittance','Etat Quittance','Type mouvement','Num avenant','Prime totale quittance']]
        dfy['paiement']= dfy.apply(lambda x : (Paiement(x['Date Encaiss quittance'],0) if np.isnan(x['Montant Encaissé quittance']) else Paiement(x['Date Encaiss quittance'],x['Montant Encaissé quittance'])),axis = 1)
        
        #travailler sur les données dupliquées:
        try:
            dfy_dup = self.df[['Num quittance','Type mouvement','Num avenant','Prime totale quittance']]
            dfy_dup  = pd.concat(g for _, g in dfy_dup.groupby("Num quittance") if len(g) > 1)
            dfy_dup = (dfy_dup.groupby(['Num quittance'])
             .apply(lambda v: dict(zip(v['Type mouvement'], zip(v['Num avenant'],v['Prime totale quittance']))))
             .reset_index(name='mapping'))
            dfy_dup['dic_TypeMouvement'] = dfy_dup.apply (lambda x: dict(zip(set(k for k,v in x['mapping'].items()),set(TypeMouvement(k,v[0],v[1]) for k,v in x['mapping'].items()))) , axis =1)
            dfyf = dfy.join (dfy_dup.set_index('Num quittance'), on='Num quittance')
        except:
            dfyf=dfy.copy()
            dfyf['dic_TypeMouvement'] = None
            
        #join les deux df
       
        
        dfyf['dic_TypeMouvement2'] = dfyf.apply(lambda x: dict({x['Type mouvement']:TypeMouvement(x['Type mouvement'],x['Num avenant'],x['Prime totale quittance'])}) if pd.isna(x['dic_TypeMouvement']) else x['dic_TypeMouvement'],axis=1) 
        
        # creation objet quittance
        dfyf['quittance'] = dfyf.apply(lambda x: Quittance(x['Num quittance'],
                                                    x['Date effet quittance'],
                                                    x['Date échéance quittance'],
                                                    x['Num police'],  
                                                    x['dic_TypeMouvement2'],
                                                    x['paiement'],
                                                    x['Etat Quittance'])
                                               ,axis=1)
        
        # creation du dict
        dfyf_dic = dfyf[['Num quittance','quittance']]
        self.collectionQuittances = dfyf_dic.set_index('Num quittance').to_dict()['quittance']
        
        return
    
    def calculStats(self):
        col = ['numQuittance','annee','mois','prime','solde' ]
        data =  [[k, v.annee,v.mois, v.prime,v.solde] for k,v in self.collectionQuittances.items()]
        df1 = pd.DataFrame(data = data ,columns = col )
            
        df1 ['paiement'] = df1 ['prime'] + df1 ['solde']
        #df1  ['tx_paiement'] = df1 ['paiement'] / df1 ['prime']
        self.totalNombre = len(df1)
        self.totalSolde = df1['solde'].sum()
        self.totalPrime = df1['prime'].sum()
        evolAnnuel = df1.groupby('annee')[["prime", "solde","paiement"]].sum()
        evolAnnuel  ['tx_paiement'] = evolAnnuel ['paiement'] / evolAnnuel ['prime']
        self.evolAnnuel = evolAnnuel.copy()
        self.evolMensuel = df1.groupby('mois')[["prime","solde","paiement"]].sum()
        
        return
    
    def evolAnnuel2(self,idx,critere):
        # idex = annee, mois ; criere = prime, solde
        res = self.evolAnnuel.loc[idx,critere] 
        return res

class Assurances:
    def __init__(self,df,collectionQuittances):
        self.collectionQuittances = collectionQuittances
        self.df = df
        
        self.collectionAssurances = dict()
        self.stats= None
        self.totalNombre = None
        self.totalPrime = None
        self.totalSolde = None
        self.dfBranche = None
        self.dfCategorie = None
        self.brancheRentable = None
        self.categorieRentable = None
        
        
     
    def initCollection(self):
        
        ##new version
        ##liste de num quittance pour chaque police
        df_numAssurance = self.df.drop_duplicates(subset = "Num police")
        df_numAssurance = df_numAssurance[['Num police',
                                  'Date effet police',
                                  'Date échéance quittance',
                                  'Libellé branche',
                                  'Libellé catégorie',
                                  'Code souscripteur'
                                 ]]


        dfx  = self.df [['Num police','Num quittance']]
        dfx = (dfx.groupby(['Num police'])
               .apply(lambda x: [x['Num quittance'].unique()])
               .reset_index(name='mapping'))
        dfx ['ls_quittance'] = dfx.apply(lambda x: x['mapping'][0].tolist(),axis=1)
        dfx.drop('mapping', axis =1, inplace = True)
        dfx = df_numAssurance.set_index('Num police').join(dfx.set_index('Num police')).reset_index() 
        dfx['objetAssurance']  = dfx.apply(lambda x: Assurance(x['Num police'],
                                             x['Date effet police'],
                                             x['Date échéance quittance'],
                                             x['Libellé branche'],
                                             x['Libellé catégorie'],
                                             x['Code souscripteur'],
                                             x['ls_quittance']) ,axis=1)
        dfx = dfx[['Num police','objetAssurance']]
        res = dfx.set_index('Num police').to_dict()['objetAssurance']
        
        self.collectionAssurances = res

         
        return

    def calculStats(self):
        col = ['numPolice', 'branche', 'categorie','prime','solde']
        data = [[k, 
                 v.branche, 
                 v.categorie, 
                 v.calculSoldeAssurance(self.collectionQuittances)[0],  
                 v.calculSoldeAssurance(self.collectionQuittances)[1]] 
                for k,v in self.collectionAssurances.items()]
           
        df1 = pd.DataFrame(data = data ,columns = col )
          
        df1.sort_values(by='solde', ascending = False, inplace = True)
        
        self.stats = df1.copy()
        self.totalNombre = len(df1)
        self.totalPrime = df1['prime'].sum()
        self.totalSolde = df1['solde'].sum()
        brancheNbDf = pd.DataFrame(df1['branche'].value_counts(normalize=False))
        branchePrime = pd.DataFrame(df1.groupby('branche')['prime'].sum()).sort_values(by='prime',ascending = False)
        brancheSolde = pd.DataFrame(df1.groupby('branche')['solde'].sum()).sort_values(by='solde',ascending = True)
        df_branche = brancheNbDf.join(branchePrime)
        df_branche = df_branche.join(brancheSolde)
        df_branche ['tx_paiement'] = (df_branche['prime']+df_branche['solde'])/df_branche['prime']
        df_branche ['ranking']= df_branche.apply(lambda x: (x['prime']*x['tx_paiement'] if x['solde']<=0 else 0), axis =1)
        self.dfBranche = df_branche.copy()


    

        dfx = df_branche.sort_values(by ='ranking', ascending = False)
        brancheBestProduct = dfx.head(1).index.tolist()[0]
        #debug
        #print(dfx.head(1)['branche'])
        #exit()
        brancheBestNb = dfx.head(1)['count'].tolist()[0]
        brancheBestNbPercentage = (dfx.head(1)['count'] / dfx['count'].sum()).tolist()[0]
        brancheBestPrime = dfx.head(1)['prime'].tolist()[0]
        brancheBestPrimePercentage = (dfx.head(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        brancheBestTxPaiement = dfx.head(1)['tx_paiement'].tolist()[0]
        brancheRentable = dict()
        brancheRentable = {'product':brancheBestProduct,
                                'nb' : brancheBestNb,
                                'nbPercentage' : brancheBestNbPercentage,
                                'prime' : brancheBestPrime,
                                'primePercentage':brancheBestPrimePercentage,
                                'txPaiement': brancheBestTxPaiement
                               }
        self.brancheRentable = brancheRentable
        
        

        categorieNbDf = pd.DataFrame(df1['categorie'].value_counts(normalize=False))
        categoriePrime = pd.DataFrame(df1.groupby('categorie')['prime'].sum()).sort_values(by='prime',ascending = False)
        categorieSolde = pd.DataFrame(df1.groupby('categorie')['solde'].sum()).sort_values(by='solde',ascending = True)
        df_categorie = categorieNbDf.join(categoriePrime)
        df_categorie = df_categorie.join(categorieSolde)
        df_categorie ['tx_paiement'] = (df_categorie['prime']+df_categorie['solde'])/df_categorie['prime']
        df_categorie ['ranking']= df_categorie.apply(lambda x: (x['prime']*x['tx_paiement'] if x['solde']<=0 else 0), axis =1)
        self.dfCategorie = df_categorie.copy()
        
        dfx = df_categorie.sort_values(by ='ranking', ascending = False)
        categorieBestProduct = dfx.head(1).index.tolist()[0]

        #debug
        categorieBestNb = dfx.head(1)['count'].tolist()[0]
        categorieBestNbPercentage = (dfx.head(1)['count'] / dfx['count'].sum()).tolist()[0]
        categorieBestPrime = dfx.head(1)['prime'].tolist()[0]
        categorieBestPrimePercentage = (dfx.head(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        categorieBestTxPaiement = dfx.head(1)['tx_paiement'].tolist()[0]
        categorieRentable = dict()
        categorieRentable = {'product':categorieBestProduct,
                                'nb' : categorieBestNb,
                                'nbPercentage' : categorieBestNbPercentage,
                                'prime' : categorieBestPrime,
                                'primePercentage':categorieBestPrimePercentage,
                                'txPaiement': categorieBestTxPaiement
                               }
        self.categorieRentable = categorieRentable
        return
        

class Clients:
    def __init__(self, df,instCollectionAssurances):
        self.collectionAssurances = instCollectionAssurances
        self.df = df
        
        self.collectionClients = dict()
        self.totalNombre = None
        self.totalSolde = None
        self.totalPrime = None
        self.dfClient = None
        self.clientPlusRentable = None
        self.client2ePlusRentable = None
        self.clientPlusGrosDeficit = None
     
        
        
    def initCollection(self):
        
        ##new version
        ##liste de num quittance pour chaque police
        df_codeClient = self.df.drop_duplicates(subset = "Code souscripteur")
        df_codeClient = df_codeClient[['Code souscripteur',
                                  'souscripteur',
                                  'Adresse Ass',
                                  'Tel Ass',
                                  'Mail Ass',
                                 ]]


        dfx  = self.df [['Code souscripteur','Num police']]
        dfx = (dfx.groupby(['Code souscripteur'])
               .apply(lambda x: [x['Num police'].unique()])
               .reset_index(name='mapping'))
        dfx ['ls_contrats'] = dfx.apply(lambda x: x['mapping'][0].tolist(),axis=1)
        dfx.drop('mapping', axis =1, inplace = True)
        dfx = df_codeClient.set_index('Code souscripteur').join(dfx.set_index('Code souscripteur')).reset_index() 
        dfx['objetClient']  = dfx.apply(lambda x: Client(int(x['Code souscripteur']),
                                                      x['souscripteur'],
                                                      x['Adresse Ass'],
                                                      x['Tel Ass'],
                                                      x['Mail Ass'], 
                                                      x['ls_contrats']),axis=1)
        dfx = dfx[['Code souscripteur','objetClient']]
        res = dfx.set_index('Code souscripteur').to_dict()['objetClient']
        self.collectionClients = res

        
        
        return
       
    def calculStats(self):
     
        col = ['numClient','nom', 'prime' ,'solde']
        data = [[k,
                 v.nom, 
                 v.calculSoldeClient(self.collectionAssurances)[0], 
                 v.calculSoldeClient(self.collectionAssurances)[1]] for k,v in self.collectionClients.items()]
        df1 = pd.DataFrame(data = data ,columns = col )
        
        self.totalSolde = df1['solde'].sum()
        self.totalNombre = len(df1)
        self.totalPrime = df1['prime'].sum()
        
        df1 ['tx_paiement'] = (df1['prime']+df1['solde'])/df1['prime']
        df1 ['ranking']= df1.apply(lambda x: (x['prime']*x['tx_paiement'] if x['solde']<=0 else 0), axis =1)
        self.dfClient = df1.copy()
        
        dfx = df1.sort_values(by ='ranking', ascending = False)
        clientPlusRentableNom  = dfx.head(1)['nom'].tolist()[0]
        clientPlusRentablePrime = dfx.head(1)['prime'].tolist()[0]
        clientPlusRentablePrimePercentage = (dfx.head(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        clientPlusRentableTxPaiement = dfx.head(1)['tx_paiement'].tolist()[0]
        clientPlusRentableSolde = dfx.head(1)['solde'].tolist()[0]
        clientPlusRentable = dict()
        clientPlusRentable = {'nom':clientPlusRentableNom,
                                'prime' : clientPlusRentablePrime,
                                'primePercentage':clientPlusRentablePrimePercentage,
                                'txPaiement': clientPlusRentableTxPaiement,
                                 'solde': clientPlusRentableSolde
                               }
        self.clientPlusRentable = clientPlusRentable

        dfx = df1.sort_values(by ='ranking', ascending = False)
        client2ePlusRentableNom  = dfx.head(2).tail(1)['nom'].tolist()[0]
        client2ePlusRentablePrime = dfx.head(2).tail(1)['prime'].tolist()[0]
        client2ePlusRentablePrimePercentage = (dfx.head(2).tail(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        client2ePlusRentableTxPaiement = dfx.head(2).tail(1)['tx_paiement'].tolist()[0]
        client2ePlusRentableSolde = dfx.head(2).tail(1)['solde'].tolist()[0]
        client2ePlusRentable = dict()
        client2ePlusRentable = {'nom':client2ePlusRentableNom,
                                'prime' : client2ePlusRentablePrime,
                                'primePercentage':client2ePlusRentablePrimePercentage,
                                'txPaiement': client2ePlusRentableTxPaiement,
                                 'solde': client2ePlusRentableSolde
                               }
        self.client2ePlusRentable = client2ePlusRentable
        
        
        
        dfx = df1.sort_values(by ='solde', ascending = True)
        clientPlusGrosDeficitNom  = dfx.head(1)['nom'].tolist()[0]
        clientPlusGrosDeficitPrime = dfx.head(1)['prime'].tolist()[0]
        clientPlusGrosDeficitPrimePercentage = (dfx.head(1)['prime'] / dfx['prime'].sum()).tolist()[0]
        clientPlusGrosDeficitTxPaiement = dfx.head(1)['tx_paiement'].tolist()[0]
        clientPlusGrosDeficitSolde = dfx.head(1)['solde'].tolist()[0]
        clientPlusGrosDeficit = dict()
        clientPlusGrosDeficit = {'nom':clientPlusGrosDeficitNom,
                                'prime' : clientPlusGrosDeficitPrime,
                                'primePercentage':clientPlusGrosDeficitPrimePercentage,
                                'txPaiement': clientPlusGrosDeficitTxPaiement,
                                 'solde': clientPlusGrosDeficitSolde
                               }
        self.clientPlusGrosDeficit = clientPlusGrosDeficit


           
       
        
   
        return

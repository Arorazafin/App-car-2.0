import dash_html_components as html
import dash_bootstrap_components as dbc


layout = dbc.Container([
            html.Br(),
            
            dbc.Row([
                dbc.Col("Agence Générale", className="h2 text-center")
            ]),

            dbc.Row([
                dbc.Col("CAR", className="h1 text-center")
            ]),
            html.Br(),

            dbc.Row([
                
                dbc.Col([
                    dbc.Card(
                        dbc.Button("Suivi ORASS-CAR (dép)",color="primary", href="/orass",),
                        body=True, 
                        color="dark", 
                        outline=True
                    )
                ],xs=12, sm=6, md=6, lg=3, xl=3),


                dbc.Col([
                    dbc.Card(
                        dbc.Button("Suivi SAGE-CAR",color="primary", href="/sage-car",),
                        body=True, 
                        color="dark", 
                        outline=True
                    )
                ],xs=12, sm=6, md=6, lg=3, xl=3),

                dbc.Col([
                    dbc.Card(
                        dbc.Button("Suivi ORASS-CAR",color="primary", href="/orassv2",),
                        body=True, 
                        color="dark", 
                        outline=True
                    )
                ],xs=12, sm=6, md=6, lg=3, xl=3),

                
                dbc.Col([
                    dbc.Card(
                        dbc.Button("...",color="primary",),
                        body=True, 
                        color="dark", 
                        outline=True
                    )
                ],xs=12, sm=6, md=6, lg=3, xl=3),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.Button("...",color="primary"),
                        body=True, 
                        color="dark", 
                        outline=True
                    )
                ],xs=12, sm=6, md=6, lg=3, xl=3),
                dbc.Col([
                    dbc.Card(
                        dbc.Button("...",color="primary"),
                        body=True, 
                        color="dark", 
                        outline=True
                    )
                ],xs=12, sm=6, md=6, lg=3, xl=3),
                dbc.Col([
                    dbc.Card(
                        dbc.Button("...",color="primary"),
                        body=True, 
                        color="dark", 
                        outline=True
                    )
                ],xs=12, sm=6, md=6, lg=3, xl=3),

                dbc.Col([
                    dbc.Card(
                        dbc.Button("...",color="primary"),
                        body=True, 
                        color="dark", 
                        outline=True
                    )
                ],xs=12, sm=6, md=6, lg=3, xl=3),
    
            ]),

        
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),


            dbc.Row([
                dbc.Col(html.A(children='Copyright 2022 CAR')
                        , className="mb-5 text-center")
            ]),
    
            

    ],fluid=True
)


# needed only if running this as a single page app
# if __name__ == '__main__':
#     app.run_server(host='127.0.0.1', debug=True)
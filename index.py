
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app
from app import server

# import all pages in the app
from apps import home 
#from apps import acnh
from apps import orass 
from apps import sage
from apps import orassv2 





print('dash version: ', dash.__version__) 

# building the navigation bar
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py



nav_item = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("Home", href="/home", id="page-1-link")),
    #    dbc.NavItem(dbc.NavLink("orass-acnh", href="/orass-acnh", id="page-2-link")),
        dbc.NavItem(dbc.NavLink("orass-car(dep)", href="/orass", id="page-2-link")),
        dbc.NavItem(dbc.NavLink("sage-car", href="/sage-car", id="page-3-link")),
        dbc.NavItem(dbc.NavLink("orass-car", href="/orassv2", id="page-4-link")),
               
            
        #dbc.NavItem(dbc.NavLink("Référence", href="/reference", id="page-3-link")),
    ],
    fill= True,
)

dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Entry 1"),
        dbc.DropdownMenuItem("Entry 2"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Entry 3"),
    ],
    nav=True,
    in_navbar=True,
    #label="Menu",
)

navbar = dbc.Navbar(
#    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                    dbc.Col(html.Img(src="/assets/nyhavana-logo.png", height="30px"), className="ml-2"),
                    ],
                    className="page-1-link",
                    align="center",
                    no_gutters=True,
                ),
                href="/home",
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            #dbc.Nav([nav_item], navbar=True, className="ml-2",)
            dbc.Collapse(
                    dbc.Nav(
                        [nav_item],
                        className="ml-2",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                    is_open=False,
            ),
            
    ],
#    ),
#    className="mb-5",
    color="#F0F0F0",
    sticky = "top"
)


@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 5)],
    [Input("url", "pathname")],
)

def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False, False
    return [pathname == f"/{i}" for i in ["home","orass-car(dep)","sage-car","orass-car"]]


# embedding the navigation bar
app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        navbar,

        dbc.Container(id="page-content", className="pt-1", fluid=True),
    ]
)


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
              
def display_page(pathname):
    if pathname == '/home' or pathname == '/':
        return home.layout       
    elif pathname == '/orass':
        return orass.layout
    elif pathname == '/orassv2':
        return orassv2.layout    
    elif pathname == '/sage-car':
        return sage.layout
    #elif pathname == '/orass-acnh':
    #    return acnh.layout    
    else:  
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-primary"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )

# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open



if __name__ == '__main__':
    #flask
    #app.run_server(debug=True)
    #app.run_server(host='0.0.0.0', debug=True, port=8050)
    app.run_server(host='0.0.0.0', debug=True, use_reloader=False)
   
    
    #run with waitress (wsgi)
    #server.run()
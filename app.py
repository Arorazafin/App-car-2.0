import dash
import dash_bootstrap_components as dbc
import dash_auth

# bootstrap theme
# https://bootswatch.com/lux/
#external_stylesheets = [dbc.themes.LUX]
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#app = dash.Dash(__name__)

# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = {
    'CAR': 'car01082021',
    #'NH':'nh123',
    #'temp':'temp123'

}

app = dash.Dash(
    __name__,
    meta_tags=[{'name': 'viewport',
               'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}],
    #meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]

)


server = app.server
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)


app.config.suppress_callback_exceptions = True
app.title = "CAR"

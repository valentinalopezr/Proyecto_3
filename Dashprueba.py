import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from dash.dependencies import Input, Output, State
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the trained model
model = tf.keras.models.load_model("modelo_saber11.h5")

# Define features and their types for preprocessing
features = {
    'cole_area_ubicacion': 'category',
    'cole_bilingue': 'category',
    'cole_calendario': 'category',
    'cole_genero': 'category',
    'estu_depto_presentacion': 'category',
    'fami_estratovivienda': 'numeric',
    'fami_tieneautomovil': 'category',
    'fami_tienecomputador': 'category',
    'fami_tieneinternet': 'category',
    'fami_tienelavadora': 'category',
    'desemp_ingles': 'numeric'
}

# Dash app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Predicción de Puntaje Saber 11"),

       html.Label('Calendario del Colegio'),
            dcc.Dropdown(
                id='cale-dropdown',
                options=[
                    {'label': 'A', 'value': 'A'},
                    {'label': 'B', 'value': 'B'}
                ],
                value=1
            ),

        html.Label('Colegio Bilingüe'),
            dcc.Dropdown(
                id='bili-dropdown',
                options=[
                    {'label': 'Sí', 'value': 'S'},
                    {'label': 'No', 'value': 'N'}
                ],
                value=1
            ),

        html.Label('Ubicación del Colegio'),
            dcc.Dropdown(
                id='ubi-dropdown',
                options=[
                    {'label': 'Rural', 'value': 'RURAL'},
                    {'label': 'Urbano', 'value': 'URBANO'}
                ],
                value=1
            ),

        html.Label('Género del Colegio'),
            dcc.Dropdown(
                id='genero-dropdown',
                options=[
                    {'label': 'Femenino', 'value': 'FEMENINO'},
                    {'label': 'Masculino', 'value': 'MASCULINO'},
                    {'label': 'Mixto', 'value': 'MIXTO'}
                ],
                value=1
            ),

        html.Label('Departamento'),
            dcc.Dropdown(
                id='dpto-dropdown',
                options=[
                    {'label': 'Antioquia', 'value': 'ANTIOQUIA'},
                    {'label': 'Bogotá', 'value': 'BOGOTÁ'}
                    
                ],
                value=1
            ),

        html.Label('Posee computador'),
            dcc.Dropdown(
                id='comp-dropdown',
                options=[
                    {'label': 'Sí', 'value': 'Si'},
                    {'label': 'No', 'value': 'No'}
                   
                ],
                value=1
            ),

        html.Label('Posee conexión a internet'),
            dcc.Dropdown(
                id='inte-dropdown',
                options=[
                    {'label': 'Sí', 'value': 'Si'},
                    {'label': 'No', 'value': 'No'}
                   
                ],
                value=1
            ),

        html.Label('Posee lavadora'),
            dcc.Dropdown(
                id='lav-dropdown',
                options=[
                    {'label': 'Sí', 'value': 'Si'},
                    {'label': 'No', 'value': 'No'}
                   
                ],
                value=1
            ),

        html.Label('Posee automovil'),
            dcc.Dropdown(
                id='aut-dropdown',
                options=[
                    {'label': 'Sí', 'value': 'Si'},
                    {'label': 'No', 'value': 'No'}
                   
                ],
                value=1
            ),

        html.Label('Estrato'),
            dcc.Dropdown(
                id='est-dropdown',
                options=[
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                    {'label': '3', 'value': 3},
                    {'label': '4', 'value': 4},
                    {'label': '5', 'value': 5},
                    {'label': '6', 'value': 6},
                   
                ],
                value=1
            ),

    html.Label('Nivel de Inglés'),
            dcc.Dropdown(
                id='ing-dropdown',
                options=[
                    {'label': 'A-', 'value': 'A-'},
                    {'label': 'A1', 'value': 'A1'},
                    {'label': 'A2', 'value': 'A2'},
                    {'label': 'B1', 'value': 'B1'},
                    {'label': 'B2', 'value': 'B2'},
                    {'label': 'B+', 'value': 'B+'},
                   
                ],
                value=1
            ),

             html.Button('Predecir', id='button', n_clicks=0),

            html.Div(id='output-prediction')
            
            ])



# Callback to handle user input and make predictions
@app.callback(
    Output('output-prediction', 'children'),
    [Input('button', 'n_clicks')],
    [dash.dependencies.State('cale-dropdown', 'value'),
    dash.dependencies.State('bili-dropdown', 'value'),
    dash.dependencies.State('ubi-dropdown', 'value'),
    dash.dependencies.State('genero-dropdown', 'value'),
    dash.dependencies.State('dpto-dropdown', 'value'),
    dash.dependencies.State('comp-dropdown', 'value'),
    dash.dependencies.State('inte-dropdown', 'value'),
    dash.dependencies.State('lav-dropdown', 'value'),
    dash.dependencies.State('aut-dropdown', 'value'),
    dash.dependencies.State('est-dropdown', 'value'),
    dash.dependencies.State('ing-dropdown', 'value')
    ]
)
def update_prediction(n_clicks, cale, bili, ubi, genero, dpto, comp, inte, lav, aut, est, ing ):
    # Preprocess user input data
    if n_clicks is not None:
        user_input = {
            'cole_calendario': [cale],
            'cole_bilingue': [bili],
            'cole_genero' : [genero],
            'cole_area_ubicacion': [ubi],
            'estu_depto_presentacion' : [dpto],
            'fami_tieneautomovil' : [aut],
            'fami_tienecomputador' : [comp],
            'fami_tieneinternet': [inte],
            'fami_tienelavadora' : [lav],
            'desemp_ingles': [ing],
            'fami_estratovivienda': [est]
        }

        pipeline = joblib.load('pipeline.pkl')
        scaler_y = joblib.load('scaler_y.pkl')

        user_input = pd.DataFrame(user_input)

        model_input = pipeline.transform(user_input)

        

    # Hacer la predicción con el modelo cargado
        prediction = model.predict(model_input).flatten()
        prediction_a = scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()

        nota = int(prediction_a[0])

        if nota > 200 and nota < 300:
            anal = 'Regular'
        
        elif nota > 300:
            anal = 'Bueno'
        
        elif nota < 250:
            anal = 'Deficiente'



        return f'Su nota predictiva es de {nota} lo cual es un desdempeño {anal}'

if __name__ == '__main__':
    app.run_server(debug=True)

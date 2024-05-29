import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from dash_table import DataTable  # Importa DataTable desde dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import tensorflow as tf
import numpy as np
import joblib

# Cargar el conjunto de datos
file_path = "C:/Users/William/Desktop/Proyecto_3/datossaber11.csv"
df = pd.read_csv(file_path)

var_usar = ['cole_bilingue', 'cole_calendario', 'cole_genero', 'estu_depto_presentacion','cole_area_ubicacion','fami_estratovivienda',
            'fami_tienelavadora', 'fami_tienecomputador', 'fami_tieneinternet', 'fami_tieneautomovil', 'desemp_ingles', 'punt_global']


df = df[var_usar]

df['cole_area_ubicacion'] = df['cole_area_ubicacion'].replace({0: 'Urbano', 1: 'Rural'})
df['cole_bilingue'] = df['cole_bilingue'].replace({0: 'No', 1: 'Sí'})
df['cole_calendario'] = df['cole_calendario'].replace({0: 'A', 1: 'B'})
df['cole_genero'] = df['cole_genero'].replace({0: 'MIXTO', 1: 'FEMENINO', 2: 'MASCULINO'})
df['estu_depto_presentacion'] = df['estu_depto_presentacion'].replace({0: 'ANTIOQUIA', 1: 'BOGOTÁ'})
df['desemp_ingles'] = df['desemp_ingles'].replace({0: 'A-', 1: 'A1', 2: 'A2', 3: 'B1', 4: 'B+'})

# Convertir las columnas categóricas a tipo 'category'
categorical_columns = ['cole_bilingue', 'cole_calendario', 'cole_genero', 'estu_depto_presentacion','cole_area_ubicacion','fami_estratovivienda',
                        'fami_tienelavadora', 'fami_tienecomputador', 'fami_tieneinternet', 'fami_tieneautomovil', 'desemp_ingles']
df[categorical_columns] = df[categorical_columns].astype('category')

model = tf.keras.models.load_model("modelo_saber11.h5")


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

# Crear la aplicación de Dash
app = dash.Dash(__name__)


group_estrat = df.groupby('fami_estratovivienda')['punt_global'].agg(['mean', 'std']).reset_index()
group_area = df.groupby('cole_area_ubicacion')['punt_global'].agg(['mean', 'std']).reset_index()

group_genero = df.groupby('cole_genero')['punt_global'].agg(['mean', 'std']).reset_index()
group_dpto = df.groupby('estu_depto_presentacion')['punt_global'].agg(['mean', 'std']).reset_index()
group_ingles = df.groupby('desemp_ingles')['punt_global'].agg(['mean', 'std']).reset_index()

# Diseño del tablero


app.layout = html.Div([
    html.H1("Análisis de Datos - Dashboard", style={'textAlign': 'center'}),

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Análisis Descriptivo', value='tab-1', children=[
            dcc.Tabs(id='tabs-1', value='tab-1-1', children=[
                dcc.Tab(label='Estrato', value='tab-1-1'),
                dcc.Tab(label='Ubicación del Colegio', value='tab-1-2'),
                dcc.Tab(label='Género del Colegio', value='tab-1-3'),
                dcc.Tab(label='Departamento de Presentación', value='tab-1-4'),
                dcc.Tab(label='Desempeño en Inglés', value='tab-1-5'),
            ]),
        ]),
        dcc.Tab(label='Predicciones', value='tab-2'),
    ]),
    html.Div(id='tabs-content', style={'padding': '20px'})
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value'),
               Input('tabs-1', 'value')])
def render_content(tab, subtab):



    if tab == 'tab-1':
        if subtab == 'tab-1-1':
            return dcc.Graph(
                id='bar-chart-estrato',
                figure=px.bar(group_estrat, x='fami_estratovivienda', y='mean',
                               error_y='std', labels={'mean': 'Puntaje Global', 'fami_estratovivienda': 'Estrato'})
            )
        elif subtab == 'tab-1-2':
            return dcc.Graph(
                id='bar-chart-ubicacion',
                figure=px.bar(group_area, x='cole_area_ubicacion', y='mean',
                               error_y='std', labels={'mean': 'Puntaje Global', 'cole_area_ubicacion': 'Ubicación del Colegio'})
            )
        elif subtab == 'tab-1-3':
            return dcc.Graph(
                id='bar-chart-genero',
                figure=px.bar(group_genero, x='cole_genero', y='mean',
                               error_y='std', labels={'mean': 'Puntaje Global', 'cole_genero': 'Género del Colegio'})
            )
        elif subtab == 'tab-1-4':
            return dcc.Graph(
                id='bar-chart-dpto',
                figure=px.bar(group_dpto, x='estu_depto_presentacion', y='mean',
                               error_y='std', labels={'mean': 'Puntaje Global', 'estu_depto_presentacion': 'Departamento de Presentación'})
            )
        elif subtab == 'tab-1-5':
            return dcc.Graph(
                id='bar-chart-ingles',
                figure=px.bar(group_ingles, x='desemp_ingles', y='mean',
                               error_y='std', labels={'mean': 'Puntaje Global', 'desemp_ingles': 'Desempeño en Inglés'})
            )
    elif tab == 'tab-2':
        return html.Div([

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

            html.Button('Predecir', id='button', style={
                'display': 'block',
                'margin': 'auto',
                'color': 'white',  
                'backgroundColor': '#007BFF',  
                'border': 'none',  
                'padding': '15px 32px',  
                'textAlign': 'center',  
                'textDecoration': 'none',  
                'fontSize': '16px',  
                'position': 'relative',
                'boxShadow': '0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23)',  
                'cursor': 'pointer',  
                'transitionDuration': '0.4s',  
                ':hover': {
                    'backgroundColor': '#0046d5',  
                    'boxShadow': '0 12px 22px rgba(0,0,0,0.29), 0 6px 6px rgba(0,0,0,0.23)',  
                }
            }),
            
            html.Div(id='output-prediction', style = {'fontsize':'24px','textAlign': 'center', 'font-weight': 'bold'})

        ])






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

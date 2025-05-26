import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import timedelta
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

# --- Parámetros ---
n_steps = 12  # Ventana temporal

# --- Cargar modelo y scaler ---

model = load_model('./model/modelo_lstm.h5', compile=False)  # cambio aquí para evitar erroeres en Railway con compile
scaler = joblib.load('./model/scaler.gz')

# --- Cargar dataset procesado ---
df = pd.read_csv('./data/datos_modelo_lstm.csv', parse_dates=['Fecha'])

# --- Inicializar app Dash ---
app = dash.Dash(__name__)
app.title = "Predicción de Consumo de Agua"

app.layout = html.Div([
    html.Img(src='/figs/ans_banner_1920x200.png', style={'width': '100%', 'height': 'auto'}),
    html.H1("Predicción de Consumo de Agua - Selecciona horizonte"),
    dcc.Slider(
        id='slider-meses',
        min=1,
        max=12,
        step=1,
        value=6,
        marks={i: f'{i} meses' for i in range(1, 13)}
    ),
    dcc.Graph(id='grafico-pronostico')
])

@app.callback(
    Output('grafico-pronostico', 'figure'),
    Input('slider-meses', 'value')
)
def actualizar_grafico(n_meses):
    X_last = scaler.transform(df.loc[:, scaler.feature_names_in_])[-n_steps:]
    forecast = []

    for _ in range(n_meses):
        x_input = X_last.reshape((1, n_steps, X_last.shape[1]))
        y_pred = model.predict(x_input, verbose=0)
        last_aux = X_last[-1, 1:]  # variables auxiliares fijas
        new_step = np.concatenate([y_pred.flatten(), last_aux])
        forecast.append(new_step[0])
        X_last = np.vstack((X_last[1:], new_step))

    forecast_array = np.array(forecast).reshape(-1, 1)
    aux_repeat = np.tile(X_last[-1, 1:], (n_meses, 1))
    forecast_scaled = np.concatenate([forecast_array, aux_repeat], axis=1)
    forecast_real = scaler.inverse_transform(forecast_scaled)[:, 0]

    last_date = df['Fecha'].max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=n_meses, freq='MS')
    forecast_df = pd.DataFrame({'Fecha': future_dates, 'Consumo_Pronosticado': forecast_real})

    return {
        'data': [
            go.Scatter(x=df['Fecha'], y=df['Consumo_Bogota'], mode='lines+markers', name='Histórico'),
            go.Scatter(x=forecast_df['Fecha'], y=forecast_df['Consumo_Pronosticado'], mode='lines+markers', name='Pronóstico')
        ],
        'layout': go.Layout(title=f'Pronóstico del Consumo de Agua - Próximos {n_meses} meses', xaxis={'title': 'Fecha'}, yaxis={'title': 'Consumo (m³)'})
    }

if __name__ == '__main__':
    app.run(debug=True)

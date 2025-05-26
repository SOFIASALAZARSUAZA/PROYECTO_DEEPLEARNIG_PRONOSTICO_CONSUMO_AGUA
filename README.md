# PROYECTO_DEEPLEARNIG_PRONOSTICO_CONSUMO_AGUA
# Predicción del Consumo de Agua en Bogotá con LSTM

Este proyecto implementa un modelo de redes neuronales tipo LSTM (Long Short-Term Memory) para predecir el consumo mensual de agua en la ciudad de Bogotá, basado en variables hidrometeorológicas e indicadores contextuales como precipitación, temperatura, humedad y eventos como cuarentenas o racionamientos.

---

## Tecnologías empleadas
- Python 3.12
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy
- Dash / Plotly
- Railway (Despliegue)

---

## Datos utilizados

- **Consumo de agua mensual** por municipio
- **Temperatura** promedio mensual (IDEAM)
- **Humedad relativa** promedio mensual (IDEAM)
- **Precipitación mensual acumulada** (IDEAM)
- **Eventos contextuales**:
  - `cuarentena`: 2020-03-25 a 2020-08-31
  - `aislamiento`: 2020-09-01 a 2022-07-01
  - `racionamiento`: 2024-04-11 a 2025-04-12

> Fuentes: [IDEAM - DHIME](http://dhime.ideam.gov.co/atencionciudadano/) y [Datos Abiertos EAAB](https://datosabiertos.bogota.gov.co/organization/eaab)

---

##  Características del modelo
- Arquitectura: 1 capa LSTM + 1 capa densa (output)
- Escalado con `MinMaxScaler`
- Ventana temporal: 12 meses
- Métricas alcanzadas:
  - RMSE: 0.17
  - MAE: 0.14
- Predicciones ajustables hasta 12 meses futuros

---

##  Estructura del repositorio

```
project/
├── app.py                  # App Dash principal
├── model/
│   ├── modelo_lstm.h5      # Modelo entrenado
│   └── scaler.gz           # Escalador MinMaxScaler
├── data/
│   └── datos_modelo_lstm.csv # Dataset procesado
├── assets/
│   └── ans_banner_1920x200.png # Imagen banner superior
├── Procfile               # Comando de arranque para Railway
├── requirements.txt       # Dependencias
└── README.md              # Este archivo
```

---

## Despliegue en Railway

1. Proyecto creado en [https://railway.app](https://railway.app)
   el link de la app

   ## https://proyectodeeplearnigpronosticoconsumoagua-production.up.railway.app/ 


##  Vista de la app

La interfaz permite seleccionar el número de meses a pronosticar, visualizar la serie histórica y el pronóstico generado por el modelo LSTM.

---

## 🙏 Agradecimientos

Este proyecto fue desarrollado como parte del curso de Deep Learning de la Universidad de los Andes, por Sofia Salazar Suaza

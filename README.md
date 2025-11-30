# Call Center Forecasting & Scheduling

Modelos avanzados de predicción de volumen de llamadas para centro de contacto usando LSTM (Deep Learning) y SARIMAX (análisis clásico de series temporales).

## 📊 Descripción

Este proyecto implementa dos enfoques complementarios para el forecasting de llamadas:

1. **LSTM (Long Short-Term Memory)**: Red neuronal recurrente para predicciones de 30 días
2. **SARIMAX**: Modelo estadístico con variables exógenas (holidays, client, iPhone)

## 🚀 Características

- **Horizonte de predicción**: 30 días
- **Window size**: 45 días históricos
- **Features**: Volumen de llamadas + codificación cíclica del día de la semana
- **Regularización**: Early stopping con patience=6
- **Visualización**: Gráficos interactivos con métricas integradas

## 📈 Métricas de Rendimiento

### Modelo LSTM
- **MAE**: 410.06 llamadas
- **MSE**: 0.0140 (normalizado)
- **MAPE**: 12.27%
- **Arquitectura**: LSTM(128) → LSTM(64) → Dense(128) → Dropout(0.2) → Dense(64) → Output(30)

### Modelo SARIMAX
- Incorpora estacionalidad semanal (period=7)
- Variables exógenas: holidays, client type, iPhone releases
- Validación con ADF test y ACF/PACF

## 📁 Estructura del Proyecto

```
forecasting and scheduling/
│
├── tensorflow.ipynb       # Notebook LSTM con entrenamiento y visualización
├── SarimaxTest.py        # Script SARIMAX con análisis estadístico
├── requirements.txt      # Dependencias del proyecto
├── README.md            # Este archivo
└── .gitignore          # Archivos excluidos de Git
```

## 🛠️ Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/TU_USUARIO/call-center-forecasting.git
cd call-center-forecasting
```

2. **Crear entorno virtual**:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

## 💻 Uso

### Notebook LSTM (Interactivo)
```bash
jupyter notebook tensorflow.ipynb
```
- Selecciona archivo Excel con datos históricos usando diálogo Tkinter
- Ejecuta todas las celdas secuencialmente
- Visualiza predicción de 30 días con métricas

### Script SARIMAX
```bash
python SarimaxTest.py
```
- Selecciona archivo Excel cuando se solicite
- Genera gráficos de descomposición, ACF, PACF
- Exporta forecasts a CSV

## 📊 Formato de Datos

Los archivos de entrada deben contener las siguientes columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `date` | datetime | Fecha del registro |
| `calls` | int | Volumen de llamadas diarias |
| `day` | str | Día de la semana (opcional) |
| `client` | int/bool | Indicador de tipo cliente (para SARIMAX) |
| `iphone` | int/bool | Indicador iPhone release (para SARIMAX) |
| `holiday` | int/bool | Indicador de festivo (para SARIMAX) |

## 🧪 Experimentos Realizados

Durante el desarrollo se probaron múltiples configuraciones:

- ✅ Window sizes: 30, 45, 60 días
- ✅ Features adicionales: lags (1, 7), rolling means (7, 14), diferencias
- ✅ Arquitecturas: LSTM simple, bidireccional, múltiples capas
- ✅ Regularización: Dropout, L2, Early stopping
- ✅ Horizonte: 15 días → 30 días

**Conclusión**: La configuración actual (45 días window, 3 features básicas, arquitectura moderada) ofrece el mejor balance precisión/simplicidad.

## 📚 Tecnologías

- **TensorFlow/Keras**: 2.15+
- **Statsmodels**: Análisis de series temporales
- **pmdarima**: Auto ARIMA
- **Pandas/NumPy**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización
- **scikit-learn**: Preprocessing y métricas

## 🔮 Mejoras Futuras

- [ ] Implementar forecasting multi-step rolling
- [ ] Agregar variables meteorológicas
- [ ] Modelo ensemble (LSTM + SARIMAX)
- [ ] API REST para predicciones en tiempo real
- [ ] Dashboard interactivo con Plotly/Dash
- [ ] Monitoreo de drift del modelo

## 👤 Autor

**Fabian**

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub

# %%
#manipulacion y procesamiento de datos
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog

#plot - graficos
import matplotlib.pyplot as plt
import seaborn as sns

#consistent plot size 
plt.rcParams['figure.figsize'] = (18, 6)

#ocultar advertencias - warnings
import warnings
warnings.filterwarnings(action='ignore' , category= DeprecationWarning)
warnings.filterwarnings(action='ignore' , category= FutureWarning)

#timeseries related imports
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf  # type: ignore
from pmdarima import auto_arima  # type: ignore

# %%
root = tk.Tk()
root.withdraw()
file = filedialog.askopenfilename(
    title="Seleccione el archivo",
    filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
)

# %%
df: pd.DataFrame = pd.read_excel(file, index_col='date', parse_dates=True)


# %%
# Weekday from 'date' (no offset); drop 'day' if present and place 'weekday' after 'date'
if df.index.name == 'date':
    df['weekday'] = df.index.day_name()
elif 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['weekday'] = df['date'].dt.day_name()
else:
    print("No 'date' column or index found to derive weekday.")

if 'day' in df.columns:
    df = df.drop(columns=['day'])

# Reorder columns to ensure 'weekday' immediately follows 'date' if both are columns
if 'date' in df.columns and 'weekday' in df.columns:
    cols = list(df.columns)
    cols.remove('weekday')
    date_pos = cols.index('date')
    cols.insert(date_pos + 1, 'weekday')
    df = df[cols]


# %%
df

# %%
title = 'serie de tiempo de llamadas diarias'
xlabel = ''
ylabel = 'Llamadas diarias'

ax = df['calls'].plot(title= title)
ax.autoscale(axis='x' , tight=True)
ax.set(xlabel=xlabel,ylabel=ylabel)

# %%
title = 'Llamadas diarias'
ylabel = 'Llamadas diarias'
xlabel = ''

ax = df['calls'].plot(figsize=(16,5), title= title)
ax.autoscale(axis='x' , tight=True)
ax.set(xlabel=xlabel,ylabel=ylabel)
for i in df[df['holiday']== 1].index:
    ax.axvline(x=i, color='k', alpha=0.4)

# %%
fig , axes = plt.subplots(1,2, figsize=(18,12))
sns.lineplot(df , x = df.index , y = 'calls' , label='calls', linestyle='-', ax= axes[0])
axes[0].set_title('Llamadas diarias')

# Rolling mean (7-day, past data only to avoid leakage)
if 'mean' in df.columns:
    df = df.drop(columns=['mean'])
df['mean'] = df['calls'].rolling(window=7, center=False, min_periods=3).mean()
sns.lineplot(df, x= df.index , y = 'mean' , label='mean', linestyle='-', linewidth=2, ax= axes[1])

# %%
sns.lineplot(df,x= df.index , y= 'calls' , label='calls' , linewidth=2,)
sns.lineplot(df,x= df.index , y= 'mean' , label='mean' , linewidth=2,)

# %%
# Ensure 'date' is DateTimeIndex with daily frequency and refresh 'weekday'
if 'date' in df.columns and df.index.name != 'date':
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.set_index('date')

# Enforce daily frequency (will introduce NaNs for missing days)
df = df.asfreq('D')

# Recompute weekday from index to ensure consistency
df['weekday'] = df.index.day_name()


# %%
# Trend and seasonality decomposition (weekly period=7)
# Use existing DateTimeIndex if already set; otherwise set from 'date' column
if isinstance(df.index, pd.DatetimeIndex):
    df_temp = df.copy()
elif 'date' in df.columns:
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
    df_temp = df_temp.set_index('date')
else:
    raise ValueError("No DateTimeIndex or 'date' column available for decomposition")

# Build continuous daily range and reindex
full_range = pd.date_range(start=df_temp.index.min(), end=df_temp.index.max(), freq='D')
df_temp = df_temp.reindex(full_range)

# Interpolate target series 'calls' if gaps
if 'calls' not in df_temp.columns:
    raise ValueError("Column 'calls' missing for decomposition")
df_temp['calls'] = df_temp['calls'].interpolate(method='linear')

# Decompose with explicit weekly period
result = seasonal_decompose(df_temp['calls'], period=7, model='additive')
result.plot()
plt.tight_layout()

# %%
#prueba de estacionariedad de dickey fuller aumentada (ADF)
from statsmodels.tsa.stattools import adfuller

# %%
def dickey_fuller(series, title= 'la prueba de dickey fuller de su conjunto de datos revela lo siguiente sobre la estacionariedad:'):
    '''this funtions takes a series and returns whether it is stationary or not.
    the result is based on the p-value of the ADF test.'''

    print(f'prueba de estacionariedad mediante la prueba de dickey fuller aumentada (ADF) para {title}')
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']

    out = pd.Series(result[0:4], index=labels)

    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value

    print(out.to_string())

    if result[1] <= 0.05:
        print('fuerte evidencia contra la hipótesis nula')
        print('rechazamos la hipótesis nula')
        print('los datos no tienen raíz unitaria y son estacionarios')
    else:
        print('debida evidencia para la hipótesis nula')
        print('no rechazamos la hipótesis nula')
        print('los datos tienen raíz unitaria y no son estacionarios')

# %%
dickey_fuller(df['calls'])

# %%
def tsplot (y, lags= None , figsize = (12, 7), style= 'bmh'): # [3]
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize= figsize)
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))

        y.plot(ax= ts_ax)
        p_value = adfuller(y)[1]
        ts_ax.set_title('grafico de analisis\n Dickey-fuller: p={0:.5f})'.format(p_value))
        plot_acf(y, lags= lags , ax= acf_ax)
        plot_pacf(y, lags= lags , ax= pacf_ax)
        plt.tight_layout()


# %%
tsplot(df['calls'].dropna(), lags=40)

# %%
#warnings.filterwarnings(action='ignore' , message= '')
auto_arima(df['calls'].dropna(), start_P= 1, start_Q=1, test= 'adf', seasonal= True, m= 7, trace= True, error_action= 'ignore', suppress_warnings= True, stepwise= True)

# %%
#Split into train and test set
len(df)

# %% [markdown]
# ## division de los datos 
# 
# * Datos de entrenamiento
# * Datos de prueba
# 
# vamos a utilizar los 77 dias para probar o evaluar el modelo.

# %%
# Date-based train/test split: last 77 days as test
# Ensure index is datetime
if not isinstance(df.index, pd.DatetimeIndex):
    raise ValueError("Index must be DateTimeIndex for date-based split.")

# Drop trailing NaNs in target before splitting (optional)
series_clean = df['calls']
cut_test = 77
if len(df) <= cut_test:
    raise ValueError("Not enough observations for requested test size.")

train = df.iloc[:-cut_test]
test = df.iloc[-cut_test:]
print(f"Train length: {len(train)} | Test length: {len(test)}")

# %%
train

# %%
test

# %% [markdown]
# Fit the Sarimax model

# %%
# Remove rows with NaN in 'calls' or 'holiday' from train before fitting
train_clean = train[['calls', 'holiday']].dropna()

model1 = SARIMAX(train_clean['calls'], order= (0,1,1), seasonal_order= (1,0,1,7), exog= train_clean[['holiday']] , enforce_invertibility= False)

#entrenamos el modelo  
resultado1 = model1.fit()

#nos mustra el resumen
resultado1.summary()

# %% [markdown]
# Obtenga los valores previstos y comparelos con valores de prueba reales

# %%
# Use forecast method instead of predict to avoid index mismatch
# forecast automatically handles the correct number of steps
pred1 = resultado1.forecast(steps=len(test), exog=test[['holiday']]).rename('SARIMAX (0,1,1)(1,0,1,7) predictions')
pred1.index = test.index  # Align index with test set
pred1

# %% [markdown]
# Plot the predicted and the real test values

# %%
title = 'Predicted vs Actuals calls'
ylabel = 'Llamadas diarias' 
xlabel = ''

# Residual diagnostics helper
resid1 = resultado1.resid
print(f"Residual mean: {resid1.mean():.4f} | Std: {resid1.std():.4f}")

ax = test['calls'].plot(legend= True , title= title)
pred1.plot(legend= True)
ax.autoscale(axis='x' , tight=True)
ax.set(xlabel=xlabel,ylabel=ylabel)
for x in test[test['holiday']== 1].index:
    ax.axvline(x=x, color='k', alpha=0.3)

# Residual ACF/PACF plots
fig, axes = plt.subplots(1,2, figsize=(12,4))
plot_acf(resid1.dropna(), lags=30, ax=axes[0])
axes[0].set_title('Residual ACF (Model1)')
plot_pacf(resid1.dropna(), lags=30, ax=axes[1])
axes[1].set_title('Residual PACF (Model1)')
plt.tight_layout()

# %% [markdown]
# Evaluate the model

# %%
from statsmodels.tools.eval_measures import rmse , mse

# %%
mse1 = mse(test['calls'], pred1)
rmse1 = rmse(test['calls'], pred1)
mae1 = (test['calls'] - pred1).abs().mean()
mape1 = ((test['calls'] - pred1).abs() / test['calls']).replace([np.inf, -np.inf], np.nan).dropna().mean() * 100

# %%
print(f"Sarima(0,1,1)(1,0,1,7) MSE:   {mse1:11.4f}")
print(f"Sarima(0,1,1)(1,0,1,7) RMSE:  {rmse1:11.4f}")
print(f"Sarima(0,1,1)(1,0,1,7) MAE:   {mae1:11.4f}")
print(f"Sarima(0,1,1)(1,0,1,7) MAPE:  {mape1:11.2f}%")


# %%
# Confidence interval forecast for Model 1
fc1 = resultado1.get_forecast(steps=len(test), exog=test[['holiday']])
mean_fc1 = fc1.predicted_mean
ci_fc1 = fc1.conf_int()

# Align indices with test set
mean_fc1.index = test.index
ci_fc1.index = test.index

# Generic handling of CI columns (take first two numeric columns)
if ci_fc1.shape[1] >= 2:
    lower = ci_fc1.iloc[:,0]
    upper = ci_fc1.iloc[:,1]
else:
    raise ValueError("Confidence interval DataFrame unexpected shape")

ax = test['calls'].plot(title='Forecast with 95% CI (Model 1)', figsize=(14,5), label='Actual')
mean_fc1.plot(ax=ax, label='Forecast')
ax.fill_between(lower.index, lower, upper, color='gray', alpha=0.3, label='95% CI')
ax.set(xlabel='', ylabel='Calls')
ax.legend()

# Optional error metrics for the CI forecast (same as point forecast)
mae_fc1 = (test['calls'] - mean_fc1).abs().mean()
mape_fc1 = ((test['calls'] - mean_fc1).abs() / test['calls']).replace([np.inf,-np.inf], np.nan).dropna().mean() * 100
print(f"CI Forecast MAE:  {mae_fc1:.4f}")
print(f"CI Forecast MAPE: {mape_fc1:.2f}%")

# %% [markdown]
# Agregamos la variable 'Holiday' 'Client' 

# %%
train

# %%
# Clean NaN values from exog variables before fitting
train_clean2 = train[['calls', 'holiday', 'client', 'iphone']].dropna()
model2 = SARIMAX(train_clean2['calls'], order= (0,1,1), seasonal_order= (1,0,1,7), exog= train_clean2[['holiday' , 'client' , 'iphone']] , enforce_invertibility= False)
resultado2 = model2.fit()
resultado2.summary()

# %%
# Clean test exog and use forecast method
test_exog2 = test[['holiday', 'client', 'iphone']].fillna(0)
pred2 = resultado2.forecast(steps=len(test), exog=test_exog2).rename('sarimax(0,1,1)(1,0,1,7) x2')
pred2.index = test.index
pred2

# %%
title = "Forecast con exogenas"
ylabel = "calls" 
xlabel = "" 

ax = test['calls'].plot(legend= True, title= title)
pred2.plot(legend = True)
ax.autoscale(axis = 'x' , tight= True)
ax.set(xlabel = xlabel , ylabel= ylabel)
for x in test[test['holiday']==1].index:
    ax.axvline(x=x, color = 'k' , alpha = 0.3)

# %%
train

# %%
# Fill NaN values in exog columns before fitting model3
train_exog3 = train[['calls', 'holiday', 'client', 'iphone', 'mean']].copy()
train_exog3['mean'] = train_exog3['mean'].bfill().ffill()
train_exog3 = train_exog3.dropna()

model3 = SARIMAX(train_exog3['calls'], order=(0,1,1), seasonal_order=(1,0,1,7), exog=train_exog3[['holiday', 'client', 'iphone', 'mean']], enforce_invertibility=False)
resultado3 = model3.fit()
resultado3.summary()

# %%
# Prepare test exog and forecast
test_exog3 = test[['holiday', 'client', 'iphone', 'mean']].copy()
test_exog3['mean'] = test_exog3['mean'].bfill().ffill()
test_exog3 = test_exog3.fillna(0)
pred3 = resultado3.forecast(steps=len(test), exog=test_exog3).rename('sarimax(0,1,1)(1,0,1,7) x3')
pred3.index = test.index
pred3


# %%
title = "Forecast con exogenas"
ylabel = "calls" 
xlabel = "" 

ax = test['calls'].plot(legend= True, title= title)
pred3.plot(legend = True)
ax.autoscale(axis = 'x' , tight= True)
ax.set(xlabel = xlabel , ylabel= ylabel)
for x in test[test['holiday']==1].index:
    ax.axvline(x=x, color = 'k' , alpha = 0.3)


# %%
# Export combined forecast results to CSV (non-destructive)
from pathlib import Path

# Validate required variables
required_vars = ['test', 'mean_fc1', 'lower', 'upper', 'pred1']
missing = [v for v in required_vars if v not in globals()]
if missing:
    print(f"Faltan variables necesarias antes de exportar: {missing}")
else:
    # Build output DataFrame
    export_df = pd.DataFrame({
        'date': test.index,
        'actual_calls': test['calls'].values,
        'forecast_model1': mean_fc1.reindex(test.index).values,
        'ci_lower_model1': lower.reindex(test.index).values,
        'ci_upper_model1': upper.reindex(test.index).values,
        'forecast_model2': pred2.reindex(test.index).values if 'pred2' in globals() else np.nan,
        'forecast_model3': pred3.reindex(test.index).values if 'pred3' in globals() else np.nan,
        'holiday': test['holiday'].values if 'holiday' in test.columns else np.nan,
        'client': test['client'].values if 'client' in test.columns else np.nan,
        'iphone': test['iphone'].values if 'iphone' in test.columns else np.nan,
        'weekday': test['weekday'].values if 'weekday' in test.columns else np.nan,
        'mean_rolling_7': test['mean'].values if 'mean' in test.columns else np.nan,
    })

    # Create export directory
    out_dir = Path('forecast_exports')
    out_dir.mkdir(exist_ok=True)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    out_file = out_dir / f'call_forecasts_{timestamp}.csv'
    export_df.to_csv(out_file, index=False)
    print(f"Archivo exportado: {out_file}")
    print(f"Filas: {len(export_df)} | Columnas: {list(export_df.columns)}")



from binance.client import Client
import pandas as pd
import datetime

import numpy as np

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import streamlit as st

import pandas as pd
import scipy.stats as stats

import plotly.graph_objects as go

import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Binance API credentials
api_key = 'nz72HfZBMuQd5iTEencCQXPc9lMOhcxoU2y7CKXEHvMl3opjjocw2SZnXfoqhhdn'
api_secret = 'HMAC'

client = Client(api_key, api_secret)


st.set_page_config(page_title='Crypto Risk Monitor', layout='wide')

symbol_list = ['ETHUSDT',
 'SOLUSDT',
 'UNIUSDT',
 'AAVEUSDT',
 'MKRUSDT',
 'LINKUSDT',
 'INJUSDT',
 'OPUSDT',
 'RNDRUSDT',
 'AKTUSDT',
 'HNTUSDT',
 'PRIMEUSDT',
 'ONDOUSDT',
 'NMRUSDT',
 'PENDLEUSDT',
 'RENDERUSDT',
 'BTCUSDT',
 'GMXUSDT',
 'IMXUSDT',
 'RUNEUSDT',
 'LUNA2USDT',
 'STXUSDT']

def returns_heatmap(df, classe):
        janelas = ['1D', '3D', '1W', '2W', '1M', '3M', '6M', '1Y']
        matriz = pd.DataFrame(columns=janelas, index=df.columns)

#         df_2y = df.ffill().pct_change(520).iloc[-1]
        df_1y = df.ffill().pct_change(260).iloc[-1]
        df_6m = df.ffill().pct_change(130).iloc[-1]
        df_3m = df.ffill().pct_change(60).iloc[-1]
        df_1m = df.ffill().pct_change(20).iloc[-1]
        df_2w = df.ffill().pct_change(10).iloc[-1]
        df_1w = df.ffill().pct_change(5).iloc[-1]
        df_3d = df.ffill().pct_change(3).iloc[-1]
        df_1d = df.ffill().pct_change(1).iloc[-1]


        matriz['1D'] = df_1d
        matriz['3D'] = df_3d
        matriz['1W'] = df_1w
        matriz['2W'] = df_2w
        matriz['1M'] = df_1m
        matriz['3M'] = df_3m
        matriz['6M'] = df_6m
        matriz['1Y'] = df_1y
#         matriz['2Y'] = df_2y
        
        matriz = matriz.dropna()
        
        annotations = []
        for y, row in enumerate(matriz.values):
            for x, val in enumerate(row):
                annotations.append({
                    "x": matriz.columns[x],
                    "y": matriz.index[y],
                    "font": {"color": "black"},
                    "text": f"{val:.2%}",
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                })
        
        fig = go.Figure(data=go.Heatmap(
                        z=matriz.values,
                        x=matriz.columns.tolist(),
                        y=matriz.index.tolist(),
                        colorscale='RdYlGn',
                        zmin=matriz.values.min(), zmax=matriz.values.max(),  # para garantir que o 0 seja neutro em termos de cor
                        hoverongaps = False,
            text=matriz.apply(lambda x: x.map(lambda y: f"{y:.2%}")),
            hoverinfo='y+x+text',
            showscale=True,
            colorbar_tickformat='.2%'
        ))
        
        
        fig.update_layout(title=classe, annotations=annotations, width=1100,  # Largura do gráfico
    height=800  # Altura do gráfico
)
        st.plotly_chart(fig)



# Define custom start and end time
start_time = datetime.datetime(2023, 1, 1, 0, 0, 0)
end_time = datetime.datetime.now()

# Initialize empty DataFrames
price_close = pd.DataFrame()

# Instantiate Binance Client (Replace with your API keys if needed)
client = Client(api_key=api_key, api_secret=api_secret, requests_params={'timeout': 30})

for symbol in symbol_list:
    try:
        klines = client.get_historical_klines(
            symbol=symbol, 
            interval=Client.KLINE_INTERVAL_1DAY,
            start_str=str(start_time), 
            end_str=str(end_time)
        )

        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(klines, columns=[
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Close Time', 'Quote Asset Volume', 'Number of Trades', 
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
        ])

        # Convert numeric columns to float
        columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Number of Trades', 
                               'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']
        df[columns_to_convert] = df[columns_to_convert].astype(float)

        # Convert timestamps
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

        # Create DataFrames with Open, Close, and Volume indexed by Close Time
        temp_df_close = pd.DataFrame({symbol: df['Close'].values}, index=df['Close Time'])


        # Concatenate to the main DataFrames
        price_close = pd.concat([price_close, temp_df_close], axis=1)
  
    
    except:
        pass



def z_score(returns, window=21):
    """
    Calcula o Z-score dos retornos de um ativo considerando uma janela de 1 mês (~21 dias úteis).
    
    :param returns: Série de retornos diários do ativo.
    :param window: Número de dias para calcular média e desvio padrão (default = 21).
    :return: Série com os valores do Z-score ao longo do tempo.
    """
    
    
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std(ddof=0)  # ddof=0 para população, use ddof=1 para amostra
    z_scores = (returns - rolling_mean) / rolling_std
    return z_scores



def compute_percentile(vol, window=252):
    """
    Computes the percentile of the last value in each column relative to the column's distribution.

    :param vol_1m: DataFrame of time series.
    :return: Series containing the percentile of the last value in each column.
    """
    
    vol = vol[-window:]
    percentiles = vol.apply(lambda col: stats.percentileofscore(col.dropna(), col.iloc[-1], kind="rank"))
    return percentiles




#Daily Change

daily_returns = price_close.pct_change()

daily_change = daily_returns.iloc[-1]

z_score_1d = round(z_score(daily_returns, window=30).iloc[-1].dropna().sort_values(), 2)


# Z-score

z_score_1w = round(z_score(daily_returns, window=7).iloc[-1].dropna().sort_values(), 2)

z_score_1m = round(z_score(daily_returns, window=30).iloc[-1].dropna().sort_values(), 2)

z_score_3m = round(z_score(daily_returns, window=90).iloc[-1].dropna().sort_values(), 2)

z_score_12m = round(z_score(daily_returns, window=360).iloc[-1].dropna().sort_values(), 2)


# Returns

returns_1m = price_close.pct_change(7).iloc[-1].dropna().sort_values()

returns_1m = price_close.pct_change(30).iloc[-1].dropna().sort_values()

returns_3m = price_close.pct_change(90).iloc[-1].dropna().sort_values()

returns_12m = price_close.pct_change(360).iloc[-1].dropna().sort_values()



# Volatility

vol_1w = daily_returns.ewm(span=7).std() * np.sqrt(365)

vol_1m = daily_returns.ewm(span=30).std() * np.sqrt(365)

vol_3m = daily_returns.ewm(span=90).std() * np.sqrt(365)

vol_12m = daily_returns.ewm(span=360).std() * np.sqrt(365)



# Vol Percentile

vp_1w = round(compute_percentile( vol_1m, window=180))

vp_1m = round(compute_percentile( vol_1m, window=360))

vp_3m = round(compute_percentile( vol_3m, window=520))

vp_12m = round(compute_percentile( vol_12m, window=len(vol_12m)))




# Criando um MultiIndex para organizar os índices acima das colunas
multi_index = pd.MultiIndex.from_tuples([
    ("Price", "Last"),
    ("Price", "Daily Change"),
    ("Price", "Z-Score"),
    
    ("Returns", "1 W"),
    ("Returns", "1 M"),
    ("Returns", "3 M"),
    ("Returns", "12 M"),
    
    ("Z-Score", "1 W"),
    ("Z-Score", "1 M"),
    ("Z-Score", "3 M"),
    ("Z-Score", "12 M"),
    
    ("Realized Volatility", "1 W"),
    ("Realized Volatility", "1 M"),
    ("Realized Volatility", "3 M"),
    ("Realized Volatility", "12 M"),
    
    ("RV Percentile", "1 W"),
    ("RV Percentile", "1 M"),
    ("RV Percentile", "3 M"),
    ("RV Percentile", "12 M"),
])

# Criando o DataFrame com MultiIndex nas colunas
risk_metrics = pd.DataFrame([
    price_close.iloc[-1],
    daily_change,
    z_score_1d,
    
    returns_1w,
    returns_1m,
    returns_3m,
    returns_12m,
    
    round(z_score_1w, 2),
    round(z_score_1m, 2),
    round(z_score_3m, 2),
    round(z_score_12m, 2),
    
    vol_1w.iloc[-1].dropna(),
    vol_1m.iloc[-1].dropna(),
    vol_3m.iloc[-1].dropna(),
    vol_12m.iloc[-1].dropna(),
    
    round(vp_1w),
    round(vp_1m),
    round(vp_3m),
    round(vp_12m)
]).T

# Atribuindo os MultiIndex às colunas
risk_metrics.columns = multi_index

risk_metrics = risk_metrics.dropna()

risk_metrics_temp = risk_metrics.copy()

cols_to_format = [
    ("Price", "Daily Change"),
    ("Returns", "1 W"), ("Returns", "1 M"), ("Returns", "3 M"), ("Returns", "12 M"),
    ("Realized Volatility", "1 W"), ("Realized Volatility", "1 M"), 
    ("Realized Volatility", "3 M"), ("Realized Volatility", "12 M")
]

for col in cols_to_format:
    risk_metrics[col] = risk_metrics[col].apply(lambda x: f"{x:.2%}")
    
    
cols_to_format = [
    ("Price", "Last"), ("Price", "Z-Score"),
    ("Z-Score", "1 W"), ("Z-Score", "1 M"), ("Z-Score", "3 M"), ("Z-Score", "12 M"),
    ("RV Percentile", "1 W"), ("RV Percentile", "1 M"), 
    ("RV Percentile", "3 M"), ("RV Percentile", "12 M")
]

for col in cols_to_format:
    risk_metrics[col] = risk_metrics[col].apply(lambda x: f"{x:.2f}")

# Definindo colunas que devem ter a linha de separação
columns_with_borders = [
    ("Price", "Z-Score"),
    ("Returns", "12 M"),
    ("Z-Score", "12 M"),
    ("Realized Volatility", "12 M"),
    ("RV Percentile", "12 M")
]

# Criando um dicionário de estilos para adicionar linhas apenas no início de cada grupo do MultiIndex
def highlight_borders(val):
    """Aplica borda direita nas colunas selecionadas."""
    border_style = "border-right: 2px solid black;"
    return border_style if val.name in columns_with_borders else ""

# Aplicando o estilo ao DataFrame
styled_df = risk_metrics.style.set_table_styles([
    {"selector": "th", "props": [("text-align", "center"), ("border-bottom", "2px solid black")]}
]).applymap(lambda x: "border-right: 2px solid black;", subset=columns_with_borders)





# Ordenando os valores de Z-Score
sorted_z_score = risk_metrics_temp[("Price", "Z-Score")].sort_values()

# Definindo cores com base nos valores de Z-Score
colors = ['red' if x < 0 else 'green' for x in sorted_z_score]

# Criando o gráfico de barras com todos os rótulos visíveis
fig = go.Figure()

fig.add_trace(go.Bar(
    x=sorted_z_score.index,
    y=sorted_z_score,
    marker_color=colors
))

# Ajustando o layout para exibir todos os rótulos
fig.update_layout(
    title="Monthly Returns Move Size",
    xaxis_title="",
    yaxis_title="Move Size",
    xaxis=dict(tickmode='linear', tickangle=-45)
)

st.plotly_chart(fig)

styled_df

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from io import BytesIO
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================================
# 1. KONFIGURATION & STAMDATA
# =========================================================================
st.set_page_config(page_title="AI Multi-Score Universal Dashboard", layout="wide")

if "current_state_key" not in st.session_state:
    st.session_state.current_state_key = None

if "scores_cache" not in st.session_state:
    st.session_state.scores_cache = {}

if "all_data" not in st.session_state:
    st.session_state.all_data = {}

FIXED_INTERVAL = "1d"
DATA_CALC_PERIOD = "2y"  # Vi henter altid 2 år i baggrunden for at sikre stabile indikatorer

STAMDATA_NAVNE = {
    # OMXC25
    'ALSYDB.CO': 'Sydbank', 'AMBU-B.CO': 'Ambu B', 'BAVA.CO': 'Bavarian Nordic',
    'CARL-B.CO': 'Carlsberg B', 'COLO-B.CO': 'Coloplast B', 'DANSKE.CO': 'Danske Bank',
    'DEMANT.CO': 'Demant', 'DSV.CO': 'DSV', 'FLS.CO': 'FLSmidth', 'GMAB.CO': 'Genmab',
    'GN.CO': 'GN Store Nord', 'ISS.CO': 'ISS', 'JYSK.CO': 'Jyske Bank',
    'MAERSK-A.CO': 'A.P. Møller - Mærsk A', 'MAERSK-B.CO': 'A.P. Møller - Mærsk B',
    'NKT.CO': 'NKT', 'NSIS-B.CO': 'Novonesis B', 'NOVO-B.CO': 'Novo Nordisk B',
    'ORSTED.CO': 'Ørsted', 'PNDORA.CO': 'Pandora', 'ROCK-B.CO': 'Rockwool B',
    'RBREW.CO': 'Royal Unibrew', 'TRYG.CO': 'Tryg', 'VWS.CO': 'Vestas Wind Systems',
    'ZEAL.CO': 'Zealand Pharma',
    
    # DAX 40
    'ADS.DE': 'Adidas AG', 'AIR.DE': 'Airbus SE', 'ALV.DE': 'Allianz SE', 
    'BAS.DE': 'BASF SE', 'BAYN.DE': 'Bayer AG', 'BMW.DE': 'BMW AG', 
    'CON.DE': 'Continental AG', '1COV.DE': 'Covestro AG', 'CBK.DE': 'Commerzbank AG',
    'DBK.DE': 'Deutsche Bank AG', 'DB1.DE': 'Deutsche Börse AG', 'DHL.DE': 'DHL Group',
    'DTE.DE': 'Deutsche Telekom AG', 'DTG.DE': 'Daimler Truck Holding', 
    'EOAN.DE': 'E.ON SE', 'FRE.DE': 'Fresenius SE', 'FME.DE': 'Fresenius Medical Care',
    'HEI.DE': 'Heidelberg Materials', 'HEN3.DE': 'Henkel Vz', 'IFX.DE': 'Infineon Technologies',
    'MBG.DE': 'Mercedes-Benz Group', 'MRK.DE': 'Merck KGaA', 'MTX.DE': 'MTU Aero Engines', 
    'MUV2.DE': 'Münchener Rück', 'PUM.DE': 'Puma SE', 'RHM.DE': 'Rheinmetall AG', 
    'RWE.DE': 'RWE AG', 'SAP.DE': 'SAP SE', 'SRT3.DE': 'Sartorius Vz', 
    'SIE.DE': 'Siemens AG', 'ENR.DE': 'Siemens Energy AG', 'SHL.DE': 'Siemens Healthineers', 
    'SY1.DE': 'Symrise AG', 'VOW3.DE': 'Volkswagen Vz', 'VNA.DE': 'Vonovia SE', 
    'QIA.DE': 'Qiagen N.V.', 'ZAL.DE': 'Zalando SE', 'HNR1.DE': 'Hannover Rück',
    
    # S&P 500
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', 'NVDA': 'NVIDIA Corp.', 
    'AMZN': 'Amazon.com Inc.', 'META': 'Meta Platforms', 'GOOGL': 'Alphabet Inc.', 
    'BRK-B': 'Berkshire Hathaway', 'LLY': 'Eli Lilly & Co.', 'AVGO': 'Broadcom Inc.', 
    'TSLA': 'Tesla Inc.',

    # NASDAQ 40
    'MSFT': 'Microsoft', 'AAPL': 'Apple', 'NVDA': 'NVIDIA', 'AMZN': 'Amazon',
    'META': 'Meta Platforms', 'GOOGL': 'Alphabet A', 'GOOG': 'Alphabet C', 
    'AVGO': 'Broadcom', 'TSLA': 'Tesla', 'COST': 'Costco',
    'AMD': 'AMD', 'NFLX': 'Netflix', 'QCOM': 'Qualcomm', 'TMUS': 'T-Mobile US', 
    'INTC': 'Intel', 'INTU': 'Intuit', 'AMGN': 'Amgen', 'TEX': 'Terex', 
    'AMAT': 'Applied Materials', 'ISRG': 'Intuitive Surgical',
    'HON': 'Honeywell', 'BKNG': 'Booking Holdings', 'VRTX': 'Vertex Pharm.', 
    'PANW': 'Palo Alto Networks', 'GILD': 'Gilead Sciences', 'REGN': 'Regeneron', 
    'LRCX': 'Lam Research', 'MELI': 'MercadoLibre', 'ADP': 'ADP', 'MU': 'Micron Tech',
    'KLAC': 'KLA Corp', 'ADI': 'Analog Devices', 'MDLZ': 'Mondelez', 
    'SNPS': 'Synopsys', 'CDNS': 'Cadence Design', 'ASML': 'ASML Holding', 
    'CSCO': 'Cisco Systems', 'MAR': 'Marriott', 'ORLY': 'O\'Reilly Auto', 
    'CTAS': 'Cintas'
}

OMXC25_TICKERS = ['ALSYDB.CO', 'AMBU-B.CO', 'BAVA.CO', 'CARL-B.CO', 'COLO-B.CO', 'DANSKE.CO', 'DEMANT.CO', 'DSV.CO', 'FLS.CO', 'GMAB.CO', 'GN.CO', 'ISS.CO', 'JYSK.CO', 'MAERSK-A.CO', 'MAERSK-B.CO', 'NKT.CO', 'NSIS-B.CO', 'NOVO-B.CO', 'ORSTED.CO', 'PNDORA.CO', 'ROCK-B.CO', 'RBREW.CO', 'TRYG.CO', 'VWS.CO', 'ZEAL.CO']
SP500_SAMPLE = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'BRK-B', 'LLY', 'AVGO', 'TSLA']
DAX_TICKERS = ['ADS.DE', 'AIR.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BMW.DE', 'CBK.DE', 'CON.DE', 
    '1COV.DE', 'DB1.DE', 'DBK.DE', 'DHL.DE', 'DTE.DE', 'DTG.DE', 'EOAN.DE', 'FME.DE', 
    'FRE.DE', 'HEI.DE', 'HEN3.DE', 'HNR1.DE', 'IFX.DE', 'MBG.DE', 'MRK.DE', 'MTX.DE', 
    'MUV2.DE', 'PUM.DE', 'QIA.DE', 'RHM.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE', 'ENR.DE', 
    'SHL.DE', 'SRT3.DE', 'SY1.DE', 'VNA.DE', 'VOW3.DE', 'ZAL.DE']
nasdaq40_tickers = [
    "MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA", "COST",
    "AMD", "NFLX", "QCOM", "TMUS", "INTC", "INTU", "AMGN", "TEX", "AMAT", "ISRG",
    "HON", "BKNG", "VRTX", "PANW", "GILD", "REGN", "LRCX", "MELI", "ADP", "MU",
    "KLAC", "ADI", "MDLZ", "SNPS", "CDNS", "ASML", "CSCO", "MAR", "ORLY", "CTAS"]

# --- SIDEBAR ---
st.sidebar.header("⚙️ Dashboard Indstillinger")
univers_valg = st.sidebar.selectbox(
    "Index eller inputmetode",
    ["OMXC25 (Danmark)", "DAX (Tyskland — Komplet 40)", "S&P 500 (Uddrag)", "NASDAQ 40", "Brugerdefineret Tickerliste"],
    index=0
)

if univers_valg == "OMXC25 (Danmark)":
    valgte_tickers = OMXC25_TICKERS
elif univers_valg == "DAX (Tyskland — Komplet 40)":
    valgte_tickers = DAX_TICKERS
elif univers_valg == "S&P 500 (Uddrag)":
    valgte_tickers = SP500_SAMPLE
elif univers_valg == "NASDAQ 40":   
    valgte_tickers = nasdaq40_tickers
else:
    bruger_input = st.sidebar.text_area("Indtast tickers (separeret med komma)", value="AAPL, NVDA, RHM.DE, NOVO-B.CO")
    valgte_tickers = [t.strip().upper() for t in bruger_input.split(",") if t.strip()]

st.sidebar.write("---")
st.sidebar.header("📅 Tidsramme")
visuel_periode = st.sidebar.select_slider(
    "Grafisk visning (Zoom)",
    options=["3m", "6m", "1y", "2y"],
    value="1y",
    key="hoved_visuel_tidsramme_slider" 
)

visuel_dage_mapping = {"3m": 90, "6m": 180, "1y": 365, "2y": 730}
valgte_visuelle_dage = visuel_dage_mapping[visuel_periode]

st.sidebar.write("---")
st.sidebar.caption("© 2026 BlueSox. All rights reserved.")
univers_navne = {STAMDATA_NAVNE.get(t, t): t for t in valgte_tickers}

# =========================================================================
# 2. DATA HENTNING & CACHING
# =========================================================================
def single_ticker_download(t, period_setting):
    try:
        df = yf.download(t, period=period_setting, interval=FIXED_INTERVAL, auto_adjust=True, progress=False)
        if not df.empty and len(df) > 5:
            if isinstance(df.columns, pd.MultiIndex):
                if t in df.columns.get_level_values(0):
                    df.columns = df.columns.get_level_values(1)
                else:
                    df.columns = df.columns.get_level_values(0)
            
            df.columns = [str(c).lower() for c in df.columns]
            
            if 'adj close' in df.columns and 'close' not in df.columns:
                df['close'] = df['adj close']
            
            if 'close' in df.columns:
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                if 'volume' in df.columns:
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                
                df = df.dropna(subset=['close'])
                if not df.empty:
                    return t, df.copy()
    except Exception as e:
        pass
    return t, None

@st.cache_data(ttl=900, show_spinner=False)
def download_data_from_api(ticker_list, period_setting):
    data_store = {}
    if not ticker_list: 
        return data_store
        
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.caption(f"Starter asynkron hentning af {len(ticker_list)} aktier...")
    
    completed = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(single_ticker_download, t, period_setting): t for t in ticker_list}
        
        for future in as_completed(futures):
            t, df = future.result()
            completed += 1
            procent = int((completed / len(ticker_list)) * 100)
            progress_bar.progress(procent)
            status_text.caption(f"Henter markedsdata: Hentet {completed}/{len(ticker_list)} aktier asynkront...")
            
            if df is not None:
                data_store[t] = df
                
    progress_bar.empty()
    status_text.empty()
    return data_store

state_key = (univers_valg,DATA_CALC_PERIOD,tuple(sorted(valgte_tickers)))

if "current_state_key" not in st.session_state or st.session_state["current_state_key"] != state_key:
    with st.spinner("Henter univers data..."):
        st.session_state["all_data"] = download_data_from_api(valgte_tickers, DATA_CALC_PERIOD)
        st.session_state["current_state_key"] = state_key
        st.session_state["scores_cache"] = {}

all_data = st.session_state["all_data"]
aktive_univers_navne = {navn: ticker for navn, ticker in univers_navne.items() if ticker in all_data}

# =========================================================================
# 3. HELPER FUNKTIONER & HJÆLPE-ALGORITMER
# =========================================================================
def beregn_obv(df):
    direction = np.sign(df['close'].diff()).fillna(0)
    return (direction * df['volume']).cumsum()

def safe_savgol(series, poly=2, max_window=11):
    x = series.dropna().values.copy()
    n = len(x)
    if n < poly + 3: return series
    window = min(max_window, n)
    if window % 2 == 0: window -= 1
    try: return pd.Series(savgol_filter(x, window_length=window, polyorder=poly), index=series.dropna().index)
    except: return series

def beregn_confidence_string(score_val):
    if score_val >= 70: return "🟢 Høj"
    elif score_val >= 40: return "🟡 Middel"
    else: return "🔴 Lav kvalitet"

def beregn_teknisk_trigger(df, rsi_col, macd_col, signal_col, squeeze_active):
    if len(df) < 5: return "Neutral"
    rsi_nu, rsi_før = df[rsi_col].iloc[-1], df[rsi_col].iloc[-2]
    macd_nu, sig_nu = df[macd_col].iloc[-1], df[signal_col].iloc[-1]
    macd_før, sig_før = df[macd_col].iloc[-2], df[signal_col].iloc[-2]
    
    if rsi_før < 45 and rsi_nu >= 45: return "🟢 Køb (RSI Momentum Opbygning)"
    if rsi_før > 75 and rsi_nu <= 75: return "🔴 Sælg (RSI Udmattelse over 75)"
    if macd_før <= sig_før and macd_nu > sig_nu: return "🟢 Køb (MACD Kryds)"
    if macd_før >= sig_før and macd_nu < sig_nu: return "🔴 Sælg (MACD Kryds)"
    if squeeze_active: return "⚡ Squeeze Breakout"
    return "Neutral"

def detect_candlestick_patterns(df):
    if len(df) < 3: return "Ingen markante mønstre fundet", 0
    
    op, hi, lo, cl = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    body = cl - op
    abs_body = np.abs(body)
    total_range = hi - lo
    
    b_nu, ab_nu, tr_nu = body[-1], abs_body[-1], total_range[-1]
    ush_nu = hi[-1] - max(op[-1], cl[-1])
    lsh_nu = min(op[-1], cl[-1]) - lo[-1]
    
    b_før, ab_før = body[-2], abs_body[-2]
    b_3d, ab_3d = body[-3], abs_body[-3]
    
    avg_body = np.mean(abs_body[-10:]) if len(abs_body) >= 10 else np.mean(abs_body)
    if avg_body == 0: avg_body = 1e-5

    if b_3d < -0.5 * avg_body and ab_før < 0.3 * avg_body and b_nu > 0.5 * avg_body:
        if cl[-1] > (op[-3] + cl[-3])/2 and max(op[-2], cl[-2]) < cl[-3] and min(op[-1], cl[-1]) > max(op[-2], cl[-2]):
            return "Morning Star (Stærkt Bullish Reversal)", 10

    if b_3d > 0.5 * avg_body and ab_før < 0.3 * avg_body and b_nu < -0.5 * avg_body:
        if cl[-1] < (op[-3] + cl[-3])/2 and min(op[-2], cl[-2]) > cl[-3] and max(op[-1], cl[-1]) < min(op[-2], cl[-2]):
            return "Evening Star (Stærkt Bearish Reversal)", -10

    if tr_nu > 0 and ush_nu > (2 * ab_nu) and lsh_nu < (0.4 * ab_nu) and hi[-1] > max(hi[-2], hi[-3]):
        if ab_nu < 1.5 * avg_body: return "Shooting Star (Bearish Reversal)", -6

    if tr_nu > 0 and lsh_nu > (2 * ab_nu) and ush_nu < (0.4 * ab_nu) and lo[-1] < min(lo[-2], lo[-3]):
        if ab_nu < 1.5 * avg_body: return "Hammer (Bullish Reversal)", 6

    if b_før < -0.5 * avg_body and b_nu > 0.5 * avg_body:
        if op[-1] < cl[-2] and cl[-1] > (op[-2] + cl[-2])/2 and cl[-1] < op[-2]:
            return "Piercing Line (Bullish Reversal)", 5

    if b_før < 0 and b_nu > 0 and cl[-1] >= op[-2] and op[-1] <= cl[-2]: return "Bullish Engulfing (Momentum vending)", 5
    if b_før > 0 and b_nu < 0 and cl[-1] <= op[-2] and op[-1] >= cl[-2]: return "Bearish Engulfing (Momentum vending)", -5

    if op[-1] > min(op[-2], cl[-2]) and cl[-1] < max(op[-2], cl[-2]) and op[-1] < max(op[-2], cl[-2]) and cl[-1] > min(op[-2], cl[-2]):
        if b_før < 0 and b_nu > 0: return "Harami (Bullish Inside Bar)", 4
        if b_før > 0 and b_nu < 0: return "Harami (Bearish Inside Bar)", -4

    if tr_nu > 0 and (ab_nu / tr_nu) < 0.10 and ab_nu < 0.2 * avg_body: return "Doji (Ubeslutsomhed / Neutral)", 1

    return "Ingen markante mønstre fundet", 0

# =========================================================================
# 4. CENTRALISERET INDIKATOR-LAG (FÆLLES DATA-MOTOR)
# =========================================================================
def beregn_alle_indikatorer(df):
    n_len = len(df)
    if df.empty or n_len < 5: return df
    df = df.copy()

    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['plus_dm'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']), np.maximum(df['high'] - df['high'].shift(), 0), 0)
    df['minus_dm'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()), np.maximum(df['low'].shift() - df['low'], 0), 0)
    delta = df["close"].diff()

    df["20ma"] = df["close"].rolling(min(20, n_len)).mean()
    df["50ma"] = df["close"].rolling(min(50, n_len)).mean()
    df["200ma"] = df["close"].rolling(200).mean() if n_len >= 200 else df["50ma"]
    
    df['std20'] = df['close'].rolling(min(20, n_len)).std()
    df['bb_upper'] = df['20ma'] + (2 * df['std20'].fillna(0))
    df['bb_lower'] = df['20ma'] - (2 * df['std20'].fillna(0))

    # --- SWING PARAMETRE (KORTERE HORISONT) ---
    gain_s = delta.where(delta > 0, 0).rolling(min(7, n_len-1)).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(min(7, n_len-1)).mean()
    df["rsi_swing"] = 100 - (100 / (1 + (gain_s / (loss_s + 1e-10))))
    
    # Stram stram MACD (8, 17, 5)
    df["ema8"] = df["close"].ewm(span=min(8, n_len)).mean()
    df["ema17"] = df["close"].ewm(span=min(17, n_len)).mean()
    df["macd_swing"] = df["ema8"] - df["ema17"]
    df["signal_swing"] = df["macd_swing"].ewm(span=min(5, n_len)).mean()
    df["macd_swing_hist"] = df["macd_swing"] - df["signal_swing"]
    df["atr_swing"] = tr.rolling(min(10, n_len-1)).mean()
    
    # Kortsigtet 10-dages ROC til Swing-sporet
    df["roc_swing"] = ((df["close"] - df["close"].shift(min(10, n_len-1))) / (df["close"].shift(min(10, n_len-1)) + 1e-10)) * 100
    
    plus_di_s = 100 * (df['plus_dm'].rolling(min(10, n_len-1)).mean() / (tr.rolling(min(10, n_len-1)).mean() + 1e-10))
    minus_di_s = 100 * (df['minus_dm'].rolling(min(10, n_len-1)).mean() / (tr.rolling(min(10, n_len-1)).mean() + 1e-10))
    df['adx_swing'] = (100 * (plus_di_s - minus_di_s).abs() / (plus_di_s + minus_di_s + 1e-10)).rolling(min(10, n_len-1)).mean()

    # --- POSITION PARAMETRE (MELLEMLANG HORISONT) ---
    gain_p = delta.where(delta > 0, 0).rolling(min(14, n_len-1)).mean()
    loss_p = (-delta.where(delta < 0, 0)).rolling(min(14, n_len-1)).mean()
    df["rsi_pos"] = 100 - (100 / (1 + (gain_p / (loss_p + 1e-10))))
    
    # Standard mellemlang MACD (12, 26, 9)
    df["ema12"] = df["close"].ewm(span=min(12, n_len)).mean()
    df["ema26"] = df["close"].ewm(span=min(26, n_len)).mean()
    df["macd_pos"] = df["ema12"] - df["ema26"]
    df["signal_pos"] = df["macd_pos"].ewm(span=min(9, n_len)).mean()
    df["macd_pos_hist"] = df["macd_pos"] - df["signal_pos"]
    df["atr_pos"] = tr.rolling(min(14, n_len-1)).mean()
    
    plus_di_p = 100 * (df['plus_dm'].rolling(min(14, n_len-1)).mean() / (tr.rolling(min(14, n_len-1)).mean() + 1e-10))
    minus_di_p = 100 * (df['minus_dm'].rolling(min(14, n_len-1)).mean() / (tr.rolling(min(14, n_len-1)).mean() + 1e-10))
    df['adx_pos'] = (100 * (plus_di_p - minus_di_p).abs() / (plus_di_p + minus_di_p + 1e-10)).rolling(min(14, n_len-1)).mean()
    
    # Mellemlang 20-dages ROC låst til positions-analyse
    df["roc_pos"] = ((df["close"] - df["close"].shift(min(20, n_len-1))) / (df["close"].shift(min(20, n_len-1)) + 1e-10)) * 100

    df["obv"] = beregn_obv(df)
    df["obv_ma10"] = df["obv"].rolling(min(10, n_len)).mean()
    df['vol_ema_swing'] = df['volume'].ewm(span=14, adjust=False).mean()
    df['vol_ma30_pos'] = df['volume'].rolling(min(30, n_len)).mean()
    df["smoothed"] = safe_savgol(df["close"])
    df["natr_pos"] = (df["atr_pos"] / (df["close"] + 1e-10)) * 100

    return df

def beregn_dual_scores(df_raw):
    if df_raw.empty or len(df_raw) < 15: return None
    
    df = beregn_alle_indikatorer(df_raw)
    
    close_price = round(float(df["close"].iloc[-1]), 1)
    vol_ratio_swing = df['volume'].iloc[-1] / (df['vol_ema_swing'].iloc[-1] + 1e-10)
    vol_ratio_pos = df['volume'].iloc[-1] / (df['vol_ma30_pos'].iloc[-1] + 1e-10)
    
    squeeze_nu = (df["close"].iloc[-1] > df["bb_upper"].iloc[-1]) and (vol_ratio_swing > 1.5)
    
    rsi_s = df["rsi_swing"].iloc[-1]
    rsi_p = df["rsi_pos"].iloc[-1]
    atr_s = df["atr_swing"].iloc[-1]
    atr_p = df["atr_pos"].iloc[-1]
    adx_s = df['adx_swing'].dropna().iloc[-1] if not df['adx_swing'].dropna().empty else 20
    adx_p = df['adx_pos'].dropna().iloc[-1] if not df['adx_pos'].dropna().empty else 20
    natr_p = df["natr_pos"].iloc[-1] if "natr_pos" in df.columns else 2.5

    candle_pattern, candle_points = detect_candlestick_patterns(df)
    
    c_ma20, c_ma50, c_ma200 = df["20ma"].iloc[-1], df["50ma"].iloc[-1], df["200ma"].iloc[-1]
    if c_ma200 and (close_price > c_ma20 > c_ma50 > c_ma200): c_trend = 100
    elif c_ma200 and (close_price > c_ma50 > c_ma200) and (close_price > c_ma20): c_trend = 85
    elif close_price > c_ma20 > c_ma50: c_trend = 75
    elif close_price > c_ma50: c_trend = 55
    elif c_ma200 and close_price > c_ma200: c_trend = 40
    elif close_price > c_ma20: c_trend = 25
    else: c_trend = 10

    conf_adx_s = min(100, max(0, int(adx_s * 2.2)))
    conf_adx_p = min(100, max(0, int(adx_p * 2.2)))
    conf_vol_swing = min(100, max(10, int(vol_ratio_swing * 60)))
    conf_vol_pos = min(100, max(10, int(vol_ratio_pos * 60)))
    
    if len(df) >= 3:
        accel_score = (df["macd_swing_hist"].iloc[-1] - df["macd_swing_hist"].iloc[-2]) * 1.0 + (df["macd_swing_hist"].iloc[-2] - df["macd_swing_hist"].iloc[-3]) * 0.5
        hist_direction_s = 100 if accel_score >= 0 else 25
    else:
        hist_direction_s = 100 if (df["macd_swing_hist"].iloc[-1] >= df["macd_swing_hist"].iloc[-2]) else 25
        
    hist_direction_p = 100 if (df["macd_pos_hist"].iloc[-1] >= df["macd_pos_hist"].iloc[-2]) else 25
    
    raw_conf_s = (conf_adx_s * 0.40) + (conf_vol_swing * 0.35) + (hist_direction_s * 0.25)
    raw_conf_p = (conf_adx_p * 0.40) + (conf_vol_pos * 0.35) + (hist_direction_p * 0.25)
    
    confidence_string_swing = beregn_confidence_string(raw_conf_s)
    confidence_string_pos = beregn_confidence_string(raw_conf_p)

    c_rsi_s = 100 if 60 <= rsi_s <= 75 else (75 if 45 <= rsi_s < 60 else (40 if 30 <= rsi_s < 45 else 15))
    c_rsi_p = 100 if 60 <= rsi_p <= 75 else (75 if 45 <= rsi_p < 60 else (40 if 30 <= rsi_p < 45 else 15))
    c_macd_s = 100 if (df["macd_swing"].iloc[-1] > df["signal_swing"].iloc[-1] and df["macd_swing_hist"].iloc[-1] > 0) else 20
    c_vol_score_s = min(100, max(0, int(vol_ratio_swing * 50)))
    c_vol_score_p = min(100, max(0, int(vol_ratio_pos * 50)))
    c_squeeze = 100 if squeeze_nu else (50 if close_price > df["20ma"].iloc[-1] else 10)
    c_adx_s = min(100, max(10, int(adx_s * 2.5)))
    
    swing_score = (c_rsi_s * 0.20 + c_macd_s * 0.20) + (c_vol_score_s * 0.20 + c_squeeze * 0.20) + (c_trend * 0.10) + (c_adx_s * 0.10)
    swing_score = max(0, min(100, swing_score + candle_points))
    
    c_macd_p = 100 if (df["macd_pos"].iloc[-1] > df["signal_pos"].iloc[-1] and df["macd_pos_hist"].iloc[-1] > 0) else 20
    c_adx_p = min(100, max(10, int(adx_p * 2.5)))
    c_obv = 100 if df["obv"].iloc[-1] > df["obv_ma10"].iloc[-1] else 20
    c_roc_p = 100 if df["roc_pos"].iloc[-1] > 0 else 20
    
    position_score = (c_trend * 0.40) + (c_adx_p * 0.15 + c_obv * 0.15 + c_roc_p * 0.10) + (c_rsi_p * 0.05 + c_macd_p * 0.05) + (c_vol_score_p * 0.10)
    position_score = max(0, min(100, position_score))

    natr_s, natr_p_pct = (atr_s / close_price) * 100, (atr_p / close_price) * 100
    swing_target_mult = (4.0 if natr_s > 3.0 else 3.5) if adx_s > 25 else 2.5
    swing_sl_mult = (2.0 if natr_s > 3.0 else 1.5) if adx_s > 25 else 1.25
    pos_target_mult = (7.0 if natr_p_pct > 3.0 else 6.0) if adx_p > 25 else 4.5
    pos_sl_mult = (3.0 if natr_p_pct > 3.0 else 2.5) if adx_p > 25 else 2.0

    return {
        "df": df, "close": close_price, "swing_score": swing_score, "position_score": position_score,
        "confidence_swing": confidence_string_swing, "confidence_pos": confidence_string_pos,
        "adx_swing": adx_s, "adx_pos": adx_p, "rsi_swing": rsi_s, "rsi_pos": rsi_p,
        "vol_ratio_swing": vol_ratio_swing, "vol_ratio_pos": vol_ratio_pos,
        "candle": candle_pattern, "squeeze": squeeze_nu, "natr_pos": natr_p,
        "swing": {
            "target": close_price + (atr_s * swing_target_mult), "sl": close_price - (atr_s * swing_sl_mult),
            "rr": ((close_price + (atr_s * swing_target_mult)) - close_price) / max(0.01, close_price - (close_price - (atr_s * swing_sl_mult))),
            "t_pct": (((close_price + (atr_s * swing_target_mult)) / close_price) - 1) * 100, "sl_pct": (1 - ((close_price - (atr_s * swing_sl_mult)) / close_price)) * 100
        },
        "position": {
            "target": close_price + (atr_p * pos_target_mult), "sl": close_price - (atr_p * pos_sl_mult),
            "rr": ((close_price + (atr_p * pos_target_mult)) - close_price) / max(0.01, close_price - (close_price - (atr_p * pos_sl_mult))),
            "t_pct": (((close_price + (atr_p * pos_target_mult)) / close_price) - 1) * 100, "sl_pct": (1 - ((close_price - (atr_p * pos_sl_mult)) / close_price)) * 100
        }
    }

# =========================================================================
# 5. STRAGTIGT RIGTIG DATO-BASERET BACKTESTING (REEL PORTEFØLJESTYRING)
# =========================================================================
def run_vectorized_backtest(data_dict, mode="Swing", min_score=75, target_mult=3.5, sl_mult=1.5, holding_days=20, start_date=None):
    aktie_data = {}
    alle_datoer = set()
    
    for ticker, base_df in data_dict.items():
        if len(base_df) < 30: continue
        
        df = beregn_alle_indikatorer(base_df)
        if start_date:
            df = df[df.index >= start_date]
        if len(df) < 2: continue
        
        if mode == "Swing":
            c_rsi = np.where((df["rsi_swing"] >= 60) & (df["rsi_swing"] <= 75), 100, np.where((df["rsi_swing"] >= 45) & (df["rsi_swing"] < 60), 75, np.where((df["rsi_swing"] >= 30) & (df["rsi_swing"] < 45), 40, 15)))
            c_macd = np.where((df["macd_swing"] > df["signal_swing"]) & (df["macd_swing_hist"] > 0), 100, 20)
            c_trend = np.where(df["close"] > df["20ma"], 75, 25)
            c_roc = np.where(df["roc_swing"] > 0, 100, 20)
            
            df["final_score"] = (c_rsi * 0.25) + (c_macd * 0.25) + (c_trend * 0.25) + (c_roc * 0.25)
            df["atr_used"] = df["atr_swing"]
        else:
            c_trend = np.where((df["close"] > df["20ma"]) & (df["20ma"] > df["50ma"]) & (df["50ma"] > df["200ma"]), 100, 30)
            c_rsi = np.where((df["rsi_pos"] >= 55) & (df["rsi_pos"] <= 75), 100, 40)
            c_macd = np.where((df["macd_pos"] > df["signal_pos"]), 100, 20)
            c_roc = np.where(df["roc_pos"] > 0, 100, 20)
            
            df["final_score"] = (c_trend * 0.40) + (c_rsi * 0.20) + (c_macd * 0.20) + (c_roc * 0.20)
            df["atr_used"] = df["atr_pos"]

        df["signal_entry"] = (df["final_score"] >= min_score) & (df["final_score"].shift(1) < min_score)
        df["atr_used"] = df["atr_used"].fillna(df["close"] * 0.02) 
        
        aktie_data[ticker] = df
        alle_datoer.update(df.index.tolist())
        
    kronologisk_tid = sorted(list(alle_datoer))
    
    start_kapital = float(bt_capital) if 'bt_capital' in locals() else 100000.0
    kontant_beholdning = start_kapital
    aktive_positioner = []  
    afsluttede_handler = []
    
    # VI SÆTTER EN FAST ALLOKERING: Hvert trade fylder 20% af din startkapital (f.eks. 20.000 kr.)
    # Det tillader op til 5 samtidige handler og fuld udnyttelse af pengene.
    FAST_TRADE_BELOEB = start_kapital * 0.20
    
    for aktuel_dag in kronologisk_tid:
        overlevende_positioner = []
        
        # 1. EXIT TJEK
        for pos in aktive_positioner:
            tk = pos["Ticker"]
            if aktuel_dag not in aktie_data[tk].index:
                overlevende_positioner.append(pos)
                continue
                
            dagens_bar = aktie_data[tk].loc[aktuel_dag]
            pos["Dage_Holdt"] += 1
            
            # Giv dit trailing stop mere elastik (tjekker ud fra lukketid frem for dagens absolutte bund for at undgå whipsaws)
            if dagens_bar["close"] > pos["Highest_High"]:
                pos["Highest_High"] = dagens_bar["close"]
                new_stop = pos["Highest_High"] - (dagens_bar["atr_used"] * pos["SL_Mult"])
                if new_stop > pos["Stop_Price"]:
                    pos["Stop_Price"] = new_stop
            
            lukket = False
            exit_pris = dagens_bar["close"]
            
            if dagens_bar["high"] >= pos["Target_Price"]:
                exit_pris = pos["Target_Price"]
                lukket = True
            elif dagens_bar["close"] <= pos["Stop_Price"]:
                exit_pris = pos["Stop_Price"]
                lukket = True
            elif pos["Dage_Holdt"] >= holding_days:
                exit_pris = dagens_bar["close"]
                lukket = True
                
            if lukket:
                brutto_retur = pos["Givet_Kapital"] * (exit_pris / pos["Entry_Price"])
                kontant_beholdning += brutto_retur  
                pnl_pct = ((exit_pris / pos["Entry_Price"]) - 1) * 100
                
                afsluttede_handler.append({
                    "Ticker": tk, "Entry Dato": pos["Entry_Dato"], "Exit Dato": aktuel_dag,
                    "Entry Pris": pos["Entry_Price"], "Exit Pris": exit_pris, "Afkast %": pnl_pct,
                    "Gevinst DKK": brutto_retur - pos["Givet_Kapital"]
                })
            else:
                overlevende_positioner.append(pos)
                
        aktive_positioner = overlevende_positioner
        
        # 2. ENTRY TJEK
        for ticker, df in aktie_data.items():
            if aktuel_dag in df.index and df.loc[aktuel_dag]["signal_entry"]:
                
                # Hvis vi har kontanter nok, bruger vi det faste beløb (så pengene reelt arbejder)
                if kontant_beholdning >= FAST_TRADE_BELOEB:
                    allokeret_indsats = FAST_TRADE_BELOEB
                elif kontant_beholdning > 5000:
                    allokeret_indsats = kontant_beholdning  # Brug resten hvis der er en sjat tilbage
                else:
                    continue
                
                if any(p["Ticker"] == ticker for p in aktive_positioner):
                    continue
                    
                row = df.loc[aktuel_dag]
                atr_val = row["atr_used"]
                if np.isnan(atr_val) or atr_val <= 0: continue
                
                natr = (atr_val / row["close"]) * 100
                dynamic_t_mult = target_mult * (1.15 if natr > 3.0 else 1.0)
                dynamic_sl_mult = sl_mult * (1.20 if natr > 3.0 else 1.0)
                
                kontant_beholdning -= allokeret_indsats 
                
                aktive_positioner.append({
                    "Ticker": ticker, "Entry_Dato": aktuel_dag, "Entry_Price": row["close"],
                    "Target_Price": row["close"] + (atr_val * dynamic_t_mult),
                    "Stop_Price": row["close"] - (atr_val * dynamic_sl_mult),
                    "Highest_High": row["close"], "SL_Mult": dynamic_sl_mult,
                    "Givet_Kapital": allokeret_indsats, "Dage_Holdt": 0
                })
                
    return pd.DataFrame(afsluttede_handler)

# =========================================================================
# 6. DASHBOARD INTERFACE & VISNING
# =========================================================================
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
tab1, tab2, tab3 = st.tabs(["📊 Oversigt", "📈 Aktie analyse", "🧪 Backtesting"])

def farv_scores(val):
    if isinstance(val, (int, float)):
        if val >= 75: color = '#2ecc71'
        elif val >= 55: color = '#27ae60'
        elif val >= 40: color = '#f39c12'
        else: color = '#e74c3c'
        return f'color: {color}; font-weight: bold;'
    return ''

# --- FANE 1: OVERVÅGNINGS-PANEL ---
with tab1:
    st.subheader(f"Aktie Screener — Univers: {univers_valg} (2 års historik)")
    rows = []
    if not all_data:
        st.warning("Ingen gyldige data fundet.")
    else:
        for t in all_data.keys():
            if t not in st.session_state["scores_cache"]:
                st.session_state["scores_cache"][t] = beregn_dual_scores(all_data[t])
            
            res_overview = st.session_state["scores_cache"][t]
            if res_overview:
                basis_navn = STAMDATA_NAVNE.get(t, t)
                
                current_natr = res_overview["natr_pos"]
                if current_natr < 2.0:
                    dynamic_vol_thresh = 1.1   
                elif current_natr > 4.0:
                    dynamic_vol_thresh = 1.4   
                else:
                    dynamic_vol_thresh = 1.2   
                
                cond1 = res_overview["position_score"] > 85
                cond2 = res_overview["adx_pos"] > 25
                cond3 = 55 < res_overview["rsi_pos"] < 75
                cond4 = res_overview["vol_ratio_pos"] > dynamic_vol_thresh
                
                betingelser_opfyldt = sum([cond1, cond2, cond3, cond4])
                
                if betingelser_opfyldt == 4:
                     visnings_navn = f"{basis_navn} 🥇"  
                elif betingelser_opfyldt == 3:
                    visnings_navn = f"{basis_navn} 🥈"  
                else:
                     visnings_navn = basis_navn
                
                rows.append({
                    "Aktiv Navn": visnings_navn, 
                    "Kurs": res_overview["close"],
                    "Swing Score (1-4 uger)": int(round(res_overview["swing_score"])),
                    "Swing Konfidens": res_overview["confidence_swing"],
                    "Position Score (1-3 mdr)": int(round(res_overview["position_score"])),
                    "Position Konfidens": res_overview["confidence_pos"]
                })
        
    if rows:
        screener_df = pd.DataFrame(rows).sort_values(by="Position Score (1-3 mdr)", ascending=False)
        tabel_col, luft_col = st.columns([3, 2])
        
        with tabel_col:
            st.dataframe(
                screener_df.style.map(farv_scores, subset=['Swing Score (1-4 uger)', 'Position Score (1-3 mdr)']),
                column_config={
                    "Aktiv Navn": st.column_config.TextColumn("Aktiv Navn", width="medium"),
                    "Kurs": st.column_config.NumberColumn("Kurs", format="%.1f", width=65),
                    "Swing Score (1-4 uger)": st.column_config.NumberColumn("Swing Score", width=50),
                    "Swing Konfidens": st.column_config.TextColumn("Swing Konfidens", width=120), 
                    "Position Score (1-3 mdr)": st.column_config.NumberColumn("Position Score", width=50),
                    "Position Konfidens": st.column_config.TextColumn("Position Konfidens", width=120)
                },
                hide_index=True,
                width='stretch'
            )
        
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            screener_df.to_excel(writer, sheet_name='AI Status', index=False)
        
        dato_stempel = datetime.date.today().strftime("%Y-%m-%d")
        st.download_button(
            label="📥 Eksporter til Excel", 
            data=buffer.getvalue(), 
            file_name=f"AktieScreener_export_{dato_stempel}.xlsx",
            mime="application/vnd.ms-excel"
        )

# --- FANE 2: INDIVIDUEL AKTIEANALYSE ---
with tab2:
    if aktive_univers_navne:
        valgt_navn = st.selectbox("Vælg aktie",sorted(list(aktive_univers_navne.keys())),key="analyse_aktiv")
        selected_ticker = aktive_univers_navne[valgt_navn]
        
        if selected_ticker not in st.session_state["scores_cache"]:
            st.session_state["scores_cache"][selected_ticker] = beregn_dual_scores(all_data[selected_ticker])
            
        res_analysis = st.session_state["scores_cache"][selected_ticker]
        
        if res_analysis:
            horisont_valg = st.radio("Horisont for risikostyring og indikatorparametre:",["Swing (1-4 uger)", "Position (1-3 mdr)"],horizontal=True,key="analyse_horisont")
            
            if "Swing" in horisont_valg:
                mat = res_analysis['swing']
                conf_visning = res_analysis['confidence_swing']
                v_ratio_visning = res_analysis['vol_ratio_swing']
                macd_col, sig_col, rsi_col = 'macd_swing', 'signal_swing', 'rsi_swing'
                aktuel_atr = res_analysis['df']['atr_swing'].iloc[-1]
            else:
                mat = res_analysis['position']
                conf_visning = res_analysis['confidence_pos']
                v_ratio_visning = res_analysis['vol_ratio_pos']
                macd_col, sig_col, rsi_col = 'macd_pos', 'signal_pos', 'rsi_pos'
                aktuel_atr = res_analysis['df']['atr_pos'].iloc[-1]

            full_df = res_analysis["df"].copy()
            visuel_filter_df = full_df.tail(valgte_visuelle_dage)
            
            trigger_status = beregn_teknisk_trigger(full_df, rsi_col, macd_col, sig_col, res_analysis['squeeze'])
            
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.06,
                subplot_titles=(f"Prisudvikling & Bollinger Bands — tidsrum: {visuel_periode}", "MACD Momentum & Histogram", "RSI Momentum Oscillator")
            )
            
            fig.add_trace(go.Candlestick(x=visuel_filter_df.index, open=visuel_filter_df['open'], high=visuel_filter_df['high'], low=visuel_filter_df['low'], close=visuel_filter_df['close'], name="OHLC"), row=1, col=1)
            fig.add_trace(go.Scatter(x=visuel_filter_df.index, y=visuel_filter_df["bb_upper"], line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=visuel_filter_df.index, y=visuel_filter_df["bb_lower"], line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=visuel_filter_df.index, y=visuel_filter_df[macd_col], line=dict(color="#0055ff")), row=2, col=1)
            fig.add_trace(go.Scatter(x=visuel_filter_df.index, y=visuel_filter_df[sig_col], line=dict(color="red")), row=2, col=1)
            fig.add_trace(go.Bar(x=visuel_filter_df.index, y=visuel_filter_df[macd_col+"_hist"], marker_color=['#2ecc71' if v >= 0 else '#e74c3c' for v in visuel_filter_df[macd_col+"_hist"]]), row=2, col=1)
            
            fig.add_trace(go.Scatter(x=visuel_filter_df.index, y=visuel_filter_df[rsi_col], line=dict(color="#9b59b6")), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#e74c3c", line_width=1.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#2ecc71", line_width=1.5, row=3, col=1)
            
            fig.update_layout(template="plotly_dark", height=730, showlegend=False, xaxis=dict(rangeslider=dict(visible=False)))
            st.plotly_chart(fig, width="stretch")
            
            st.write("---")
            st.subheader("🎯 Risikostyring & Taktiske Nøgletal")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Swing Score", f"{int(round(res_analysis['swing_score']))} / 100")
            c2.metric("Position Score", f"{int(round(res_analysis['position_score']))} / 100")
            c3.metric("Signal Confidence", conf_visning)

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Nuværende Pris", f"{res_analysis['close']:.1f}")
            c6.metric("Kurstarget", f"{mat['target']:.1f}", f"+{mat['t_pct']:.1f}%")
            c7.metric("Stop Loss", f"{mat['sl']:.1f}", f"-{mat['sl_pct']:.1f}%")
            c8.metric("Risk/Reward", f"1 : {mat['rr']:.2f}")
            
            st.write("")
            vol_col, atr_col, _ = st.columns([1, 1, 2])
            
            with vol_col:
                st.metric(label="Volumen Ratio", value=f"{v_ratio_visning:.2f}x")
            with atr_col:
                valuta = "DKK" if selected_ticker.endswith(".CO") else ("EUR" if selected_ticker.endswith(".DE") else "USD")
                st.metric(label="ATR (Average True Range)", value=f"{aktuel_atr:.2f} {valuta}")
            
            st.write("---")
            st.markdown(f"📊 **Candlestick Mønster:** `{res_analysis['candle']}`")
            st.write("") 
            
            if "🟢" in trigger_status: st.success(f"Taktisk Signal: {trigger_status} | Squeeze: {'⚡ Aktiv' if res_analysis['squeeze'] else 'Stabil'}")
            elif "🔴" in trigger_status: st.error(f"Taktisk Signal: {trigger_status} | Squeeze: {'⚡ Aktiv' if res_analysis['squeeze'] else 'Stabil'}")
            else: st.info(f"Taktisk Signal: {trigger_status} | Squeeze: {'⚡ Aktiv' if res_analysis['squeeze'] else 'Stabil'}")

# --- FANE 3: STRATEGI-BACKTESTING ---
with tab3:
    st.subheader("🧪 Historisk Backtesting & Strategi Optimering")
    
    cc1, cc2, cc3 = st.columns(3)
    bt_mode = cc1.selectbox("Strategi-spor til test/optimering",["Swing", "Position"],key="bt_mode")
    bt_min_score = cc2.slider("Minimum Position Score for Købs-entry", 50, 90, 80, 5)
    bt_capital = cc3.number_input("Startkapital (DKK)", value=100000, step=10000)
    
    cc4, cc5, cc6 = st.columns(3)
    bt_target_mult = cc4.slider("ATR Target Multiplikator", 1.5, 8.0, 3.5, 0.5)
    bt_sl_mult = cc5.slider("ATR Stop Loss Multiplikator", 1.0, 4.0, 2.0, 0.25)
    bt_hold = cc6.slider("Max hold periode (Handelsdage)", 5, 60, 20, 5)
    
    col_btn1, col_btn2 = st.columns(2)
    run_normal = col_btn1.button("🚀 Kør Enkel Historisk Backtest")
    run_opt = col_btn2.button("🔍 Kør Parameter Optimering")
    
    backtest_start_date = datetime.datetime.now() - datetime.timedelta(days=valgte_visuelle_dage)
    
    if run_normal:
        with st.spinner("Kværner historisk data..."):
            trades_df = run_vectorized_backtest(all_data, mode=bt_mode, min_score=bt_min_score, target_mult=bt_target_mult, sl_mult=bt_sl_mult, holding_days=bt_hold, start_date=backtest_start_date)
            
            if trades_df.empty:
                st.warning("Ingen historiske handler matchede dine kriterier i den valgte tidsperiode.")
            else:
                total_trades = len(trades_df)
                winning_trades = trades_df[trades_df["Afkast %"] > 0]
                win_rate = (len(winning_trades) / total_trades) * 100
                total_gain = winning_trades["Gevinst DKK"].sum()
                total_loss = abs(trades_df[trades_df["Afkast %"] <= 0]["Gevinst DKK"].sum())
                profit_factor = total_gain / (total_loss if total_loss > 0 else 1.0)
                
                trades_df = trades_df.sort_values(by="Entry Dato").reset_index(drop=True)
                equity_curve = [bt_capital]
                for g in trades_df["Gevinst DKK"]:
                    equity_curve.append(equity_curve[-1] + g)
                
                final_equity = equity_curve[-1]
                total_return_pct = ((final_equity / bt_capital) - 1) * 100
                eq_series = pd.Series(equity_curve)
                max_dd = ((eq_series - eq_series.cummax()) / eq_series.cummax()).min() * 100
                
                avg_return_pct = trades_df["Afkast %"].mean()

                st.markdown("### 📊 Testresultater")
                kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
                kpi1.metric("Antal Handler", f"{total_trades}")
                kpi2.metric("Win Rate", f"{win_rate:.1f}%")
                kpi3.metric("Gns. pr. Trade", f"{avg_return_pct:+.2f}%")  
                kpi4.metric("Profit Factor", f"{profit_factor:.2f}")
                kpi5.metric("Slutkapital", f"{final_equity:,.0f} DKK", f"{total_return_pct:+.1f}%")
                kpi6.metric("Max Drawdown", f"{max_dd:.1f}%")
                
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=list(range(len(equity_curve))), y=equity_curve, line=dict(color="#2ecc71", width=2.5)))
                fig_eq.update_layout(title="Egenkapitaludvikling (Reel Porteføljestyring)", template="plotly_dark", height=350)
                st.plotly_chart(fig_eq, width="stretch")

    if run_opt:
        with st.spinner("🤖 Kører Grid-Search optimering..."):
            target_range = [2.5, 3.0, 3.5, 4.0, 4.5]
            stop_range = [1.25, 1.5, 1.75, 2.0]
            opt_results = []
            
            for t_mult in target_range:
                for s_mult in stop_range:
                    # RETTET HER: s_mult er ændret til sl_mult=s_mult
                    t_df = run_vectorized_backtest(all_data, mode=bt_mode, min_score=bt_min_score, target_mult=t_mult, sl_mult=s_mult, holding_days=bt_hold, start_date=backtest_start_date)
                    
                    if not t_df.empty:
                        t_trades = len(t_df)
                        t_win_rate = (len(t_df[t_df["Afkast %"] > 0]) / t_trades) * 100
                        t_gain = t_df[t_df["Afkast %"] > 0]["Gevinst DKK"].sum()
                        t_loss = abs(t_df[t_df["Afkast %"] <= 0]["Gevinst DKK"].sum())
                        t_pf = t_gain / (t_loss if t_loss > 0 else 1.0)
                        
                        eq = bt_capital + t_df["Gevinst DKK"].sum()
                        ret_pct = ((eq / bt_capital) - 1) * 100
                        
                        opt_results.append({
                            "Target Mult": t_mult, "Stop Mult": s_mult, "Antal Trades": t_trades,
                            "Win Rate": f"{t_win_rate:.1f}%", "Profit Factor": round(t_pf, 2),
                            "Slutafkast %": round(ret_pct, 1), "Slutkapital DKK": int(eq)
                        })
            
            if not opt_results:
                st.error("Kunne ikke optimere. Ingenting matchede dine kriterier i dette tidsinterval.")
            else:
                opt_df = pd.DataFrame(opt_results).sort_values(by="Profit Factor", ascending=False)
                st.success("🎯 Optimering fuldført!")
                vinder = opt_df.iloc[0]
                st.info(f"🏆 **ANBEFALET OPSÆTNING:** Sæt dit Target til **{vinder['Target Mult']}x ATR** og dit Stop Loss til **{vinder['Stop Mult']}x ATR**. Det gav en Profit Factor på **{vinder['Profit Factor']}** under trailing-betingelser.")
                st.dataframe(opt_df, column_config={"Slutkapital DKK": st.column_config.NumberColumn("Slutkapital DKK", format="%d DKK")}, hide_index=True, width="stretch")


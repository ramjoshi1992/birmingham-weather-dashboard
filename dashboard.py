import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import folium
from streamlit_folium import st_folium
import openmeteo_requests
import requests_cache
from retry_requests import retry
import ssl
import urllib3
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Page configuration — this must be the first Streamlit command
st.set_page_config(
    page_title="Birmingham Weather Dashboard",
    page_icon="🌤️",
    layout="wide"       # uses the full browser width
)
LAT    = 52.4862
LON    = -1.8904
YEARS  = [2020, 2021, 2022, 2023, 2024]
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

PANELS = [
    ("temperature", "Temperature (°C)", "tomato",       "coral"),
    ("rainfall",    "Rainfall (mm)",     "steelblue",    "royalblue"),
    ("windspeed",   "Wind speed (km/h)", "seagreen",     "limegreen"),
    ("humidity",    "Humidity (%)",      "mediumpurple", "violet"),
]

def get_season(month):
    if month in [3, 4, 5]:    return "Spring"
    elif month in [6, 7, 8]:  return "Summer"
    elif month in [9, 10, 11]:return "Autumn"
    else:                      return "Winter"

def weather_description(code):
    code = int(code) if not pd.isna(code) else 0
    descriptions = {
        0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy",
        3:"Overcast", 45:"Foggy", 51:"Light drizzle",
        53:"Drizzle", 55:"Heavy drizzle", 61:"Slight rain",
        63:"Rain", 65:"Heavy rain", 71:"Slight snow",
        73:"Snow", 80:"Slight showers", 81:"Showers",
        82:"Heavy showers", 95:"Thunderstorm"
    }
    for key in sorted(descriptions.keys(), reverse=True):
        if code >= key:
            return descriptions[key]
    return "Unknown"

def weather_emoji(code):
    code = int(code) if not pd.isna(code) else 0
    if code == 0:    return "☀️"
    elif code <= 2:  return "🌤️"
    elif code <= 3:  return "☁️"
    elif code <= 48: return "🌫️"
    elif code <= 67: return "🌧️"
    elif code <= 77: return "❄️"
    elif code <= 82: return "🌦️"
    else:            return "⛈️"
# @st.cache_data tells Streamlit to remember the result of this function
# so it does not re-download every time someone clicks something
# ttl=3600 means the cache expires after 1 hour

@st.cache_data(ttl=3600)
def fetch_data(start_date, end_date, api_url):
    """Fetches hourly weather data for Birmingham."""
    cache_session = requests_cache.CachedSession(
        '.cache', expire_after=3600
    )
    cache_session.verify = False
    retry_session = retry(
        cache_session, retries=5, backoff_factor=0.2
    )
    retry_session.verify = False
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude":  LAT,
        "longitude": LON,
        "hourly": ["temperature_2m", "precipitation",
                   "windspeed_10m",  "relativehumidity_2m"],
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   "Europe/London"
    }

    responses = openmeteo.weather_api(api_url, params=params)
    r = responses[0]
    hourly = r.Hourly()

    df = pd.DataFrame({
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature": hourly.Variables(0).ValuesAsNumpy(),
        "rainfall":    hourly.Variables(1).ValuesAsNumpy(),
        "windspeed":   hourly.Variables(2).ValuesAsNumpy(),
        "humidity":    hourly.Variables(3).ValuesAsNumpy(),
    })

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/London")
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df.set_index("datetime", inplace=True)
    return df

@st.cache_data(ttl=3600)
def load_all_years():
    """Downloads all historical years."""
    result = {}
    for year in YEARS:
        result[year] = fetch_data(
            f"{year}-01-01",
            f"{year}-12-31",
            "https://archive-api.open-meteo.com/v1/archive"
        )
    return result
def plot_yearly(df, year):
    daily = df.resample("D").agg({
        "temperature": "mean",
        "rainfall":    "sum",
        "windspeed":   "mean",
        "humidity":    "mean"
    })
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Birmingham — full year overview {year}",
                 fontsize=14)
    for ax, (col, ylabel, colour, _) in zip(axes, PANELS):
        ax.plot(daily.index, daily[col],
                color=colour, linewidth=1.2)
        ax.fill_between(daily.index, daily[col],
                        alpha=0.15, color=colour)
        mean_val = daily[col].mean()
        ax.axhline(mean_val, color=colour, linewidth=1,
                   linestyle="--", alpha=0.7,
                   label=f"Mean: {mean_val:.1f}")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    plt.tight_layout()
    return fig

def plot_monthly(df, year):
    monthly = df.groupby(df.index.month).agg({
        "temperature": ["mean", "min", "max"],
        "rainfall":    "sum",
        "windspeed":   ["mean", "max"],
        "humidity":    "mean"
    })
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Birmingham — monthly summary {year}",
                 fontsize=14)
    axes = axes.flatten()
    mp = range(1, 13)

    ax = axes[0]
    ax.fill_between(mp,
                    monthly["temperature"]["min"],
                    monthly["temperature"]["max"],
                    alpha=0.2, color="tomato", label="Min–max")
    ax.plot(mp, monthly["temperature"]["mean"],
            color="tomato", linewidth=2.5,
            marker="o", markersize=5, label="Mean")
    ax.axhline(0, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.5)
    ax.set_title("Temperature (°C)")
    ax.set_xticks(list(mp))
    ax.set_xticklabels(MONTHS)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bars = ax.bar(mp, monthly["rainfall"]["sum"],
                  color="steelblue", alpha=0.75, width=0.6)
    for bar, val in zip(bars, monthly["rainfall"]["sum"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{val:.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("Total rainfall (mm)")
    ax.set_xticks(list(mp))
    ax.set_xticklabels(MONTHS)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    ax.bar(mp, monthly["windspeed"]["max"],
           color="lightgreen", alpha=0.5,
           width=0.6, label="Max")
    ax.bar(mp, monthly["windspeed"]["mean"],
           color="seagreen", alpha=0.85,
           width=0.6, label="Mean")
    ax.set_title("Wind speed (km/h)")
    ax.set_xticks(list(mp))
    ax.set_xticklabels(MONTHS)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[3]
    ax.plot(mp, monthly["humidity"]["mean"],
            color="mediumpurple", linewidth=2.5,
            marker="o", markersize=5)
    ax.fill_between(mp, 60, monthly["humidity"]["mean"],
                    alpha=0.2, color="mediumpurple")
    ax.set_title("Mean humidity (%)")
    ax.set_xticks(list(mp))
    ax.set_xticklabels(MONTHS)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_forecast(df_recent, df_forecast, today):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"Birmingham — 30-day history + 7-day forecast\n"
        f"Generated: {today.strftime('%d %B %Y')}",
        fontsize=13
    )
    for ax, (col, ylabel, c_hist, c_fcast) in zip(axes, PANELS):
        ax.plot(df_recent.index, df_recent[col],
                color=c_hist, linewidth=1.3, label="Observed")
        ax.fill_between(df_recent.index, df_recent[col],
                        alpha=0.12, color=c_hist)
        ax.plot(df_forecast.index, df_forecast[col],
                color=c_fcast, linewidth=2,
                linestyle="--", label="Forecast")
        ax.fill_between(df_forecast.index, df_forecast[col],
                        alpha=0.12, color=c_fcast)
        ax.axvline(x=today, color="black",
                   linewidth=1.2, linestyle=":", alpha=0.7)
        ax.axvspan(today, df_forecast.index[-1],
                   alpha=0.04, color="gray")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )
    axes[-1].xaxis.set_major_locator(
        mdates.WeekdayLocator(byweekday=0)
    )
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig

def build_map(latest):
    m = folium.Map(
        location=[LAT, LON],
        zoom_start=12,
        tiles="CartoDB positron"
    )
    popup_html = f"""
    <div style="font-family:Arial; width:260px;">
        <h3 style="color:#2C5F8A; margin-bottom:4px;">
            Birmingham, UK
        </h3>
        <p style="color:#666; font-size:11px; margin-top:0;">
            {latest.name.strftime('%d %b %Y %H:%M')}
        </p>
        <table style="font-size:13px; width:100%;">
            <tr>
                <td>🌡️ <b>Temperature</b></td>
                <td>{latest['temperature']:.1f}°C</td>
            </tr>
            <tr>
                <td>💧 <b>Humidity</b></td>
                <td>{latest['humidity']:.0f}%</td>
            </tr>
            <tr>
                <td>🌧️ <b>Rainfall</b></td>
                <td>{latest['rainfall']:.1f} mm</td>
            </tr>
            <tr>
                <td>💨 <b>Wind speed</b></td>
                <td>{latest['windspeed']:.1f} km/h</td>
            </tr>
        </table>
    </div>
    """
    folium.Marker(
        location=[LAT, LON],
        popup=folium.Popup(
            folium.IFrame(popup_html, width=280, height=200),
            max_width=300
        ),
        tooltip="Birmingham — click for weather",
        icon=folium.Icon(
            color="blue", icon="cloud", prefix="fa"
        )
    ).add_to(m)
    return m
def main():
    today     = datetime.today()
    yesterday = today - timedelta(days=1)

    # --- Sidebar controls ---
    st.sidebar.title("🌤️ Birmingham Weather")
    st.sidebar.markdown("---")
    selected_tab = st.sidebar.radio(
        "View",
        ["Current conditions",
         "Yearly overview",
         "Monthly breakdown",
         "Forecast",
         "Map"]
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Data: Open-Meteo  \n"
        f"Updated: {today.strftime('%d %b %Y')}"
    )

    # --- Load data with a spinner so user knows it is working ---
    with st.spinner("Loading weather data..."):
        data = load_all_years()
        df_recent = fetch_data(
            (today - timedelta(days=30)).strftime("%Y-%m-%d"),
            yesterday.strftime("%Y-%m-%d"),
            "https://archive-api.open-meteo.com/v1/archive"
        )
        df_forecast = fetch_data(
            today.strftime("%Y-%m-%d"),
            (today + timedelta(days=7)).strftime("%Y-%m-%d"),
            "https://api.open-meteo.com/v1/forecast"
        )

    latest = df_recent.dropna().iloc[-1]

    # -------------------------------------------------------
    # Tab: Current conditions
    # -------------------------------------------------------
    if selected_tab == "Current conditions":
        st.title("Birmingham Weather Dashboard")
        st.caption(
            f"Last updated: "
            f"{latest.name.strftime('%d %B %Y at %H:%M')}"
        )

        # Four metric cards across the top
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡️ Temperature",
                    f"{latest['temperature']:.1f}°C")
        col2.metric("💧 Humidity",
                    f"{latest['humidity']:.0f}%")
        col3.metric("💨 Wind speed",
                    f"{latest['windspeed']:.1f} km/h")
        col4.metric("🌧️ Rainfall",
                    f"{latest['rainfall']:.1f} mm")

        st.markdown("---")

        # Extreme events table
        st.subheader("Extreme weather events — 2020 to 2024")
        rows = []
        for year in YEARS:
            df   = data[year]
            dmax = df.resample("D").max()
            dmin = df.resample("D").min()
            dsum = df["rainfall"].resample("D").sum()
            rows.append({
                "Year":        year,
                "Hot days >25°C":   int((dmax["temperature"] > 25).sum()),
                "Frost days <0°C":  int((dmin["temperature"] < 0).sum()),
                "Heavy rain >10mm": int((dsum > 10).sum()),
                "High wind >50km/h":int((dmax["windspeed"] > 50).sum()),
            })
        st.dataframe(
            pd.DataFrame(rows).set_index("Year"),
            use_container_width=True
        )

    # -------------------------------------------------------
    # Tab: Yearly overview
    # -------------------------------------------------------
    elif selected_tab == "Yearly overview":
        st.title("Yearly overview")
        year = st.selectbox(
            "Select year", YEARS, index=len(YEARS)-1
        )
        fig = plot_yearly(data[year], year)
        st.pyplot(fig)
        plt.close()

    # -------------------------------------------------------
    # Tab: Monthly breakdown
    # -------------------------------------------------------
    elif selected_tab == "Monthly breakdown":
        st.title("Monthly breakdown")
        year = st.selectbox(
            "Select year", YEARS, index=len(YEARS)-1
        )
        fig = plot_monthly(data[year], year)
        st.pyplot(fig)
        plt.close()

    # -------------------------------------------------------
    # Tab: Forecast
    # -------------------------------------------------------
    elif selected_tab == "Forecast":
        st.title("30-day history + 7-day forecast")
        fig = plot_forecast(df_recent, df_forecast, today)
        st.pyplot(fig)
        plt.close()

        # Daily forecast table
        st.subheader("7-day forecast summary")
        daily_fc = df_forecast.resample("D").agg({
            "temperature": ["min", "mean", "max"],
            "rainfall":    "sum",
            "windspeed":   ["mean", "max"],
            "humidity":    "mean"
        }).round(1)
        daily_fc.index = daily_fc.index.strftime("%a %d %b")
        st.dataframe(daily_fc, use_container_width=True)

    # -------------------------------------------------------
    # Tab: Map
    # -------------------------------------------------------
    elif selected_tab == "Map":
        st.title("Birmingham weather map")
        st.caption("Click the marker for current conditions")
        m = build_map(latest)
        st_folium(m, width=900, height=500)

# --- Entry point ---
# This line runs the main() function when Streamlit starts the app
if __name__ == "__main__":
    main()

---

Save `dashboard.py`, `requirements.txt` and `packages.txt` in your `weather_dashboard` folder.

---

## Test it locally first

Before uploading to GitHub, test it on your own computer. In Anaconda Prompt:

conda activate weather_dash
pip install streamlit streamlit-folium
cd C:\Users\%USERNAME%\Documents\weather_dashboard
streamlit run dashboard.py

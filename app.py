# app.py
import re
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

DATA_DIR = Path("data")

# ---------- Helpers ----------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.astype(str)
    cols = cols.str.strip().str.lower()
    cols = cols.str.replace(r'[\s]+', '_', regex=True)
    cols = cols.str.replace(r'[^0-9a-z_]', '', regex=True)
    cols = cols.str.replace(r'__+', '_', regex=True)
    cols = cols.str.strip('_')
    df.columns = cols
    return df

def find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for key in keywords:
        for c in df.columns:
            if key in c:
                return c
    return None

def load_and_clean(path: Path, parse_date=True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    df = clean_columns(df)
    if parse_date and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# ---------- Load data ----------
try:
    facebook = load_and_clean(DATA_DIR / "Facebook.csv")
    google   = load_and_clean(DATA_DIR / "Google.csv")
    tiktok   = load_and_clean(DATA_DIR / "TikTok.csv")
    business = load_and_clean(DATA_DIR / "Business.csv")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ---------- Add platform tag ----------
facebook['platform'] = 'Facebook'
google['platform']   = 'Google'
tiktok['platform']   = 'TikTok'
marketing = pd.concat([facebook, google, tiktok], ignore_index=True, sort=False)

# ---------- Detect columns ----------
impr_col  = find_col(marketing, ["impression", "impressions"])
click_col = find_col(marketing, ["clicks", "click"])
spend_col = find_col(marketing, ["spend", "cost"])
attr_col  = find_col(marketing, ["attributed", "attributed_revenue", "attributedrevenue", "attributedrev"])

orders_col        = next((c for c in business.columns if "order" in c and "new" not in c), None)
new_orders_col    = next((c for c in business.columns if "new" in c and "order" in c), None)
new_customers_col = find_col(business, ["new_customer", "newcustomers", "new_customers", "newcustomer"])
total_rev_col     = find_col(business, ["total_revenue", "totalrevenue", "revenue", "totalrev"])
gross_col         = find_col(business, ["gross_profit", "grossprofit", "gross"])
cogs_col          = find_col(business, ["cog", "cogs"])

missing = []
if impr_col is None: missing.append("impressions")
if click_col is None: missing.append("clicks")
if spend_col is None: missing.append("spend")
if attr_col is None: missing.append("attributed_revenue")
if total_rev_col is None: missing.append("total_revenue (business)")
if missing:
    st.error(f"Missing column(s): {missing}")
    st.stop()

# ---------- Normalize columns ----------
marketing = marketing.rename(columns={
    impr_col: 'impressions',
    click_col: 'clicks',
    spend_col: 'spend',
    attr_col: 'attributed_revenue'
})
marketing[['impressions','clicks','spend','attributed_revenue']] = marketing[['impressions','clicks','spend','attributed_revenue']].apply(pd.to_numeric, errors='coerce').fillna(0)

business_renames = {}
if orders_col: business_renames[orders_col] = 'orders'
if new_orders_col: business_renames[new_orders_col] = 'new_orders'
if new_customers_col: business_renames[new_customers_col] = 'new_customers'
if total_rev_col: business_renames[total_rev_col] = 'total_revenue'
if gross_col: business_renames[gross_col] = 'gross_profit'
if cogs_col: business_renames[cogs_col] = 'cogs'
business = business.rename(columns=business_renames)
for c in ['orders','new_orders','new_customers','total_revenue','gross_profit','cogs']:
    if c not in business.columns: business[c] = 0
    business[c] = pd.to_numeric(business[c], errors='coerce').fillna(0)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Marketing Intelligence Dashboard", layout="wide")
st.title("Marketing Intelligence Dashboard")
st.markdown("This dashboard combines marketing (Facebook, Google, TikTok) and business data to provide actionable insights.")

# ---------- Sidebar ----------
min_date = business['date'].min()
max_date = business['date'].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
platforms = marketing['platform'].unique().tolist()
sel_platforms = st.sidebar.multiselect("Select Platform(s)", platforms, default=platforms)
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# ---------- Filter data ----------
marketing_f = marketing[(marketing['date'] >= start) & 
                        (marketing['date'] <= end) & 
                        (marketing['platform'].isin(sel_platforms))].copy()
marketing_daily_f = marketing_f.groupby('date', as_index=False).agg({
    'impressions':'sum','clicks':'sum','spend':'sum','attributed_revenue':'sum'
})
business_f = business[(business['date'] >= start) & (business['date'] <= end)].copy()
combined_f = pd.merge(business_f, marketing_daily_f, on='date', how='left')
combined_f[['impressions','clicks','spend','attributed_revenue']] = combined_f[['impressions','clicks','spend','attributed_revenue']].fillna(0)

# ---------- Derived metrics ----------
combined_f['ctr'] = np.where(combined_f['impressions']>0, combined_f['clicks']/combined_f['impressions'], 0)
combined_f['cpc'] = np.where(combined_f['clicks']>0, combined_f['spend']/combined_f['clicks'], 0)
combined_f['roas'] = np.where(combined_f['spend']>0, combined_f['attributed_revenue']/combined_f['spend'], 0)
combined_f['aov'] = np.where(combined_f['orders']>0, combined_f['total_revenue']/combined_f['orders'], 0)
combined_f['cac'] = np.where(combined_f['new_customers']>0, combined_f['spend']/combined_f['new_customers'], np.nan)
combined_f['margin_pct'] = np.where(combined_f['total_revenue']>0, combined_f['gross_profit']/combined_f['total_revenue'], np.nan)

# ---------- KPIs ----------
total_spend = marketing_f['spend'].sum()
total_revenue = combined_f['total_revenue'].sum()
total_roas = total_revenue / total_spend if total_spend > 0 else 0
avg_ctr = (marketing_f['clicks'].sum() / marketing_f['impressions'].sum()) if marketing_f['impressions'].sum() > 0 else 0
avg_cac = combined_f['cac'].mean()
avg_margin = combined_f['margin_pct'].mean()

with st.container():
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Spend ($)", f"{total_spend:,.0f}")
    k2.metric("Total Revenue ($)", f"{total_revenue:,.0f}")
    k3.metric("ROAS", f"{total_roas:.2f}")
    k4.metric("CTR", f"{avg_ctr:.2%}")
    k5.metric("Average CAC ($)", f"{avg_cac:.2f}")
    k6.metric("Average Margin (%)", f"{avg_margin:.2%}")

# ---------- Expandable Sections ----------
with st.expander("Time Series Analysis"):
    fig1 = px.line(combined_f, x='date', y=['spend','total_revenue'],
                   labels={'value':'Amount ($)','date':'Date'},
                   color_discrete_sequence=['#4C78A8','#F58518'],
                   title="Spend vs Revenue Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(combined_f, x='date', y=['ctr','roas'],
                   labels={'value':'Value','date':'Date'},
                   color_discrete_sequence=['#54A24B','#B279A2'],
                   title="CTR & ROAS Over Time")
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("Platform Analysis"):
    channel_metrics = marketing_f.groupby('platform').agg({
        'impressions':'sum','clicks':'sum','spend':'sum','attributed_revenue':'sum'
    }).reset_index()
    channel_metrics['ctr'] = np.where(channel_metrics['impressions']>0, channel_metrics['clicks']/channel_metrics['impressions'], 0)
    channel_metrics['roas'] = np.where(channel_metrics['spend']>0, channel_metrics['attributed_revenue']/channel_metrics['spend'], 0)
    fig3 = px.bar(channel_metrics, x='platform', y='spend', text='roas',
                  labels={'spend':'Spend ($)','platform':'Platform','roas':'ROAS'},
                  color='platform', color_discrete_sequence=['#4C78A8','#54A24B','#F58518'],
                  title="Spend by Platform")
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("Platform Efficiency"):
    platform_metrics = marketing_f.groupby('platform').agg({'spend':'sum','attributed_revenue':'sum','clicks':'sum','impressions':'sum'}).reset_index()
    total_orders = combined_f['orders'].sum()
    total_new_customers = combined_f['new_customers'].sum()
    total_spend_sum = platform_metrics['spend'].sum()
    platform_metrics['orders'] = np.where(total_spend_sum>0, platform_metrics['spend']/total_spend_sum*total_orders,0)
    platform_metrics['new_customers'] = np.where(total_spend_sum>0, platform_metrics['spend']/total_spend_sum*total_new_customers,0)
    platform_metrics['ctr'] = np.where(platform_metrics['impressions']>0, platform_metrics['clicks']/platform_metrics['impressions'],0)
    platform_metrics['roas'] = np.where(platform_metrics['spend']>0, platform_metrics['attributed_revenue']/platform_metrics['spend'],0)
    platform_metrics['aov'] = np.where(platform_metrics['orders']>0, platform_metrics['attributed_revenue']/platform_metrics['orders'],0)
    platform_metrics['cac'] = np.where(platform_metrics['new_customers']>0, platform_metrics['spend']/platform_metrics['new_customers'], np.nan)

    fig4 = px.bar(platform_metrics, x='platform', y='roas', text='aov',
                  labels={'roas':'ROAS','platform':'Platform','aov':'AOV ($)'},
                  color='platform', color_discrete_sequence=['#4C78A8','#54A24B','#F58518'],
                  title="Platform Efficiency: ROAS & AOV")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.scatter(platform_metrics, x='cac', y='roas', size='aov', color='platform',
                      hover_data=['ctr','impressions','clicks','spend','attributed_revenue'],
                      labels={'cac':'CAC ($)','roas':'ROAS','aov':'AOV ($)'},
                      color_discrete_sequence=['#4C78A8','#54A24B','#F58518'],
                      title="CAC vs ROAS per Platform (Bubble Size = AOV)")
    st.plotly_chart(fig5, use_container_width=True)

with st.expander("Conversion Funnel"):
    funnel_df = combined_f[['impressions','clicks','orders']].sum().reset_index()
    funnel_df.columns = ['stage','count']
    fig6 = px.funnel(funnel_df, x='count', y='stage', color_discrete_sequence=['#4C78A8'],
                     title="Impressions → Clicks → Orders")
    st.plotly_chart(fig6, use_container_width=True)

with st.expander("Top Campaigns"):
    campaign_metrics = marketing_f.groupby(['platform','campaign']).agg({
        'spend':'sum','attributed_revenue':'sum','clicks':'sum','impressions':'sum'
    }).reset_index()
    campaign_metrics['roas'] = np.where(campaign_metrics['spend']>0, campaign_metrics['attributed_revenue']/campaign_metrics['spend'],0)
    campaign_metrics['ctr'] = np.where(campaign_metrics['impressions']>0, campaign_metrics['clicks']/campaign_metrics['impressions'],0)
    top_campaigns = campaign_metrics.sort_values('attributed_revenue', ascending=False).head(10)
    fig_top = px.bar(top_campaigns, x='campaign', y='attributed_revenue', color='platform', text='roas',
                     labels={'attributed_revenue':'Attributed Revenue ($)','campaign':'Campaign','roas':'ROAS'},
                     color_discrete_sequence=['#4C78A8','#54A24B','#F58518'],
                     title="Top 10 Campaigns by Attributed Revenue")
    st.plotly_chart(fig_top, use_container_width=True)


from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

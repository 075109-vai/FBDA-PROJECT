"""
ShopEase Orders — Dynamic Analytical Dashboard
FBDA Project | Group ID: 095_105_107_109
Group: Prakriti Sharma (075095), Shreyash Satpathy (075105),
       Sushant (075107), Vaibhav Pratap Singh (075109)

Run locally:   streamlit run app.py
Deploy free:   push this repo to GitHub -> https://share.streamlit.io -> New app
               (point it at app.py; add shopease_sample_2500.csv to the repo,
               or keep the Kaggle-download block below active)
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="ShopEase Analytics Dashboard", layout="wide")

SEED = 95105107109
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------
# Data loading + cleaning (self-contained: mirrors data_cleaning.py
# so this single file can be deployed on its own)
# ----------------------------------------------------------------
CANON_CATEGORY = ['Electronics', 'Fashion', 'Beauty', 'Sports', 'Home']
CANON_CITY = ['Cairo', 'Giza', 'Alexandria', 'Mansoura', 'Tanta', 'Aswan']
CANON_PAYMENT = ['Cash', 'Visa', 'Wallet', 'Mastercard', 'Bank Transfer']
CANON_STATUS = ['Delivered', 'Pending', 'Cancelled']


def _norm_categorical(series, canon_list):
    s = series.astype(str).str.strip().str.title()
    s = s.replace({'Canceled': 'Cancelled'})
    lookup = {c.title(): c for c in canon_list}
    return s.map(lambda x: lookup.get(x, x))


def _parse_mixed_dates(series):
    def parse_one(v):
        if pd.isna(v):
            return pd.NaT
        v = str(v).strip()
        for fmt in ('%Y-%m-%d %H:%M:%S', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y'):
            try:
                return pd.to_datetime(v, format=fmt)
            except (ValueError, TypeError):
                continue
        return pd.to_datetime(v, errors='coerce', dayfirst=True)
    return series.apply(parse_one)


def _parse_discount(series):
    def parse_one(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        if v.endswith('%'):
            return float(v[:-1]) / 100.0
        return float(v)
    return series.apply(parse_one)


@st.cache_data
def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    df['CustomerID'] = df['CustomerID'].fillna('Unknown').astype(str).str.strip()
    df['Product'] = df['Product'].fillna('Unknown').astype(str).str.strip().str.title()
    df['OrderDate'] = _parse_mixed_dates(df['OrderDate'])
    df['DeliveryDate'] = _parse_mixed_dates(df['DeliveryDate'])
    invalid_age = (df['CustomerAge'] < 0) | (df['CustomerAge'] > 100)
    df.loc[invalid_age, 'CustomerAge'] = np.nan
    df['CustomerAge'] = df['CustomerAge'].fillna(df['CustomerAge'].median())
    g = df['Gender'].astype(str).str.strip().str.upper()
    df['Gender'] = g.map(lambda x: 'Male' if x in ('M', 'MALE') else ('Female' if x in ('F', 'FEMALE') else x.title()))
    df['City'] = _norm_categorical(df['City'], CANON_CITY)
    df['Category'] = _norm_categorical(df['Category'], CANON_CATEGORY)
    df['PaymentMethod'] = _norm_categorical(df['PaymentMethod'], CANON_PAYMENT).fillna('Unknown')
    df['OrderStatus'] = _norm_categorical(df['OrderStatus'], CANON_STATUS)
    df['Quantity'] = df['Quantity'].abs()
    df = df[df['Quantity'] != 0].reset_index(drop=True)
    df['UnitPrice'] = df['UnitPrice'].abs()
    prod_median_price = df.groupby('Product')['UnitPrice'].transform('median')
    is_outlier = df['UnitPrice'] > (prod_median_price * 5)
    df.loc[is_outlier, 'UnitPrice'] = prod_median_price[is_outlier]
    df['Discount'] = _parse_discount(df['Discount']).fillna(0.0).clip(0, 1)
    invalid_rating = df['Rating'].notna() & (~df['Rating'].isin([1, 2, 3, 4, 5]))
    df.loc[invalid_rating, 'Rating'] = np.nan
    df['TotalAmount'] = (df['UnitPrice'] * df['Quantity'] * (1 - df['Discount'])).round(2)
    df['DeliveryDays'] = (df['DeliveryDate'] - df['OrderDate']).dt.total_seconds() / 86400.0
    df['OrderMonth'] = df['OrderDate'].dt.to_period('M').astype(str)
    df['IsDelivered'] = (df['OrderStatus'] == 'Delivered').astype(int)

    rng = np.random.default_rng(SEED)
    df_sample = df.sample(n=min(2500, len(df)), random_state=rng).reset_index(drop=True)
    return df_sample


DATA_PATH = "shopease_sample_2500.csv"  # pre-sampled file shipped alongside app.py
try:
    df = pd.read_csv(DATA_PATH, parse_dates=['OrderDate', 'DeliveryDate'])
except FileNotFoundError:
    st.warning("shopease_sample_2500.csv not found — falling back to raw file + on-the-fly cleaning.")
    df = load_and_clean("shopease_raw_orders.csv")

# ----------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------
st.sidebar.title("🛒 ShopEase Dashboard")
st.sidebar.caption("Group ID: 095_105_107_109")

categories = st.sidebar.multiselect("Category", sorted(df['Category'].unique()), default=sorted(df['Category'].unique()))
cities = st.sidebar.multiselect("City", sorted(df['City'].unique()), default=sorted(df['City'].unique()))
statuses = st.sidebar.multiselect("Order Status", sorted(df['OrderStatus'].unique()), default=sorted(df['OrderStatus'].unique()))
genders = st.sidebar.multiselect("Gender", sorted(df['Gender'].unique()), default=sorted(df['Gender'].unique()))

age_min, age_max = int(df['CustomerAge'].min()), int(df['CustomerAge'].max())
age_range = st.sidebar.slider("Customer Age", age_min, age_max, (age_min, age_max))

f = df[
    df['Category'].isin(categories) & df['City'].isin(cities) &
    df['OrderStatus'].isin(statuses) & df['Gender'].isin(genders) &
    df['CustomerAge'].between(*age_range)
]

st.title("ShopEase Raw Orders — Dynamic Analytical Dashboard")
st.caption(f"{len(f):,} of {len(df):,} orders shown after filters | Fixed sample seed = {SEED}")

# ----------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Orders", f"{len(f):,}")
c2.metric("Total Revenue", f"{f['TotalAmount'].sum():,.0f}")
c3.metric("Avg Order Value", f"{f['TotalAmount'].mean():,.1f}" if len(f) else "—")
c4.metric("Avg Rating", f"{f['Rating'].mean():.2f}" if f['Rating'].notna().any() else "—")
c5.metric("Delivery Rate", f"{f['IsDelivered'].mean()*100:.1f}%" if len(f) else "—")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔗 Relationships", "🧪 Statistical Tests", "📋 Data"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orders by Category")
        st.bar_chart(f['Category'].value_counts())
        st.subheader("Payment Method Share")
        fig, ax = plt.subplots()
        f['PaymentMethod'].value_counts().plot(kind='pie', autopct='%1.1f%%', ylabel='', ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Total Amount Distribution")
        fig, ax = plt.subplots()
        sns.histplot(f['TotalAmount'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
        st.subheader("Avg Order Value by Month")
        monthly = f.groupby('OrderMonth')['TotalAmount'].mean().sort_index()
        st.line_chart(monthly)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Unit Price vs Total Amount")
        fig, ax = plt.subplots()
        sns.scatterplot(data=f, x='UnitPrice', y='TotalAmount', hue='Category', alpha=.6, s=25, ax=ax)
        st.pyplot(fig)
        st.subheader("Total Amount by Category")
        fig, ax = plt.subplots()
        sns.boxplot(data=f, x='Category', y='TotalAmount', ax=ax)
        plt.xticks(rotation=30)
        st.pyplot(fig)
    with col2:
        st.subheader("Correlation Heatmap")
        num_cols = ['CustomerAge', 'Quantity', 'UnitPrice', 'Discount', 'TotalAmount', 'Rating']
        fig, ax = plt.subplots()
        sns.heatmap(f[num_cols].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
        st.pyplot(fig)
        st.subheader("Customer Age by Gender")
        fig, ax = plt.subplots()
        sns.violinplot(data=f, x='Gender', y='CustomerAge', ax=ax)
        st.pyplot(fig)

with tab3:
    st.subheader("Live Hypothesis Tests (recompute on filtered data)")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**t-test: Total Amount, Male vs Female**")
        m = f.loc[f['Gender'] == 'Male', 'TotalAmount'].dropna()
        fem = f.loc[f['Gender'] == 'Female', 'TotalAmount'].dropna()
        if len(m) > 1 and len(fem) > 1:
            t_stat, p = stats.ttest_ind(m, fem, equal_var=False)
            st.write(f"t = {t_stat:.3f}, p = {p:.4f} → " + ("**Significant**" if p < 0.05 else "Not significant"))
        st.markdown("**ANOVA: Total Amount across Category**")
        groups = [g['TotalAmount'].dropna().values for _, g in f.groupby('Category') if len(g) > 1]
        if len(groups) > 1:
            fstat, p = stats.f_oneway(*groups)
            st.write(f"F = {fstat:.3f}, p = {p:.4f} → " + ("**Significant**" if p < 0.05 else "Not significant"))
    with colB:
        st.markdown("**Chi-square: Category vs Order Status**")
        ct = pd.crosstab(f['Category'], f['OrderStatus'])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            st.write(f"χ² = {chi2:.3f}, dof = {dof}, p = {p:.4f} → " + ("**Associated**" if p < 0.05 else "No significant association"))
        st.markdown("**Pearson correlation: Unit Price vs Total Amount**")
        sub = f[['UnitPrice', 'TotalAmount']].dropna()
        if len(sub) > 2:
            r, p = stats.pearsonr(sub['UnitPrice'], sub['TotalAmount'])
            st.write(f"r = {r:.3f}, p = {p:.4g}")

with tab4:
    st.subheader("Filtered Data")
    st.dataframe(f, use_container_width=True)
    st.download_button("Download filtered data as CSV", f.to_csv(index=False), "shopease_filtered.csv", "text/csv")

st.sidebar.divider()
st.sidebar.caption("FBDA Project — Dynamic Analytical Dashboard with Python")

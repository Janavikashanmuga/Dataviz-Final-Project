import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Global Coffee Quality Dashboard",
    page_icon="☕",
    layout="wide"
)

# 1. Load Data (Using the verified raw URL so it deploys perfectly to Streamlit Cloud)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/fatih-boyar/coffee-quality-data-CQI/main/df_arabica_clean.csv"
    data = pd.read_csv(url)
    # Basic cleaning
    data['Altitude'] = pd.to_numeric(data['Altitude'], errors='coerce')
    data = data[(data['Altitude'].notna()) & (data['Altitude'] < 3000)]
    data = data.dropna(subset=['Total Cup Points'])
    return data

df = load_data()

# 2. Application Header
st.title("☕ The Global Coffee Footprint")
st.markdown("### *An Interactive Exploration of Sensory Quality, Processing, and Geography*")
st.markdown("---")

# 3. Sidebar Filter Panel (Interactivity!)
st.sidebar.header("Filter Options")

# Country Multi-select
all_countries = sorted(df['Country of Origin'].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries of Origin",
    options=all_countries,
    default=['Ethiopia', 'Guatemala', 'Taiwan', 'Brazil', 'Colombia']
)

# Processing Method Filter
all_methods = sorted(df['Processing Method'].dropna().unique())
selected_methods = st.sidebar.multiselect(
    "Select Processing Methods",
    options=all_methods,
    default=all_methods
)

# Apply Sidebar Filters to Data
filtered_df = df[
    (df['Country of Origin'].isin(selected_countries)) &
    (df['Processing Method'].isin(selected_methods))
]

# Quick metrics on top
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Samples Selected", len(filtered_df))
with col2:
    st.metric("Avg Quality Score", f"{filtered_df['Total Cup Points'].mean():.2f} / 100")
with col3:
    st.metric("Highest Elevation", f"{filtered_df['Altitude'].max():.0f} m")

st.markdown("---")

# 4. Multi-Tab Layout (Extra Credit Design!)
tab1, tab2, tab3 = st.tabs(["🌍 Geography & Altitude", "👃 Sensory Chemistry", "⚠️ Defect Analysis"])

# --- TAB 1: GEOGRAPHY & ALTITUDE ---
with tab1:
    st.subheader("How Geography & Elevation Shape Quality")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Visual 1: Altitude vs. Quality Scatter (Question 1) - Warning Free
        plot_df = filtered_df.copy()
        highest_score = plot_df['Total Cup Points'].max() if len(plot_df) > 0 else 0
        plot_df['Highlight'] = np.where(plot_df['Total Cup Points'] == highest_score, 'Highest Quality', 'Other Coffees')
        
        fig1 = px.scatter(
            plot_df,
            x='Altitude',
            y='Total Cup Points',
            color='Highlight',
            color_discrete_map={'Other Coffees': '#b0b0b0', 'Highest Quality': '#D4AF37'},
            opacity=0.8,
            trendline="ols" if len(plot_df) > 2 else None,
            trendline_color_override="#555555",
            hover_data=['Country of Origin', 'Variety'],
            labels={'Altitude': 'Altitude (m)', 'Total Cup Points': 'Cup Score'}
        )
        fig1.update_layout(
            title="<b>Elevation Premium:</b> Sensory Scores Peak Near 2,000m",
            showlegend=False,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        # Visual 2: Altitude Spread by Country (Question 11)
        fig11 = px.box(
            filtered_df,
            x='Country of Origin',
            y='Altitude',
            color='Country of Origin',
            color_discrete_sequence=['#8c6239', '#a67c52', '#c49a6c', '#d9b48f', '#b0b0b0'],
            points="outliers",
            labels={'Altitude': 'Altitude (m)'}
        )
        fig11.update_layout(
            title="<b>Geographic Sweet Spots:</b> Elevation Spans by Country",
            showlegend=False,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig11, use_container_width=True)

# --- TAB 2: SENSORY CHEMISTRY ---
with tab2:
    st.subheader("Sensory Profile Deep-Dive")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Visual 3: Acidity vs. Body Trade-off (Question 7)
        fig7 = px.scatter(
            filtered_df,
            x='Acidity',
            y='Body',
            color='Processing Method',
            color_discrete_map={'Washed / Wet': '#c49a6c', 'Natural / Dry': '#8c6239'},
            opacity=0.7,
            labels={'Acidity': 'Acidity Score', 'Body': 'Body Score'}
        )
        fig7.update_layout(
            title="<b>Sensory Trade-off:</b> Acidity vs. Body Dynamics",
            plot_bgcolor="white"
        )
        st.plotly_chart(fig7, use_container_width=True)
        
    with col_right:
        # Visual 4: Predictors of Overall Quality (Question 8)
        sensory_cols = ['Aroma', 'Flavor', 'Acidity', 'Body', 'Balance', 'Sweetness']
        if len(filtered_df) > 1:
            correlations = filtered_df[sensory_cols].corrwith(filtered_df['Total Cup Points']).reset_index()
            correlations.columns = ['Sensory Attribute', 'Correlation']
            correlations = correlations.sort_values(by='Correlation', ascending=True)
            
            strongest_predictor = correlations.loc[correlations['Correlation'].idxmax(), 'Sensory Attribute']
            correlations['Highlight'] = correlations['Sensory Attribute'].apply(
                lambda x: 'Strongest Predictor' if x == strongest_predictor else 'Other Attributes'
            )
            
            fig8 = px.bar(
                correlations,
                x='Correlation',
                y='Sensory Attribute',
                orientation='h',
                color='Highlight',
                color_discrete_map={'Other Attributes': '#d3d3d3', 'Strongest Predictor': '#D4AF37'},
                labels={'Correlation': 'Correlation with Total Cup Points'}
            )
            fig8.update_layout(
                title="<b>The Quality Blueprint:</b> Strongest Sensory Predictor",
                showlegend=False,
                plot_bgcolor="white"
            )
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.info("Select more data options in the sidebar to view correlation matrices.")

# --- TAB 3: DEFECT ANALYSIS ---
with tab3:
    st.subheader("Defect Metrics & Processing")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Visual 5: Regional Defect Profiles (Question 2) - Guaranteed Data Coverage
        def classify_region(country):
            african = ['Ethiopia', 'Kenya', 'Uganda', 'Tanzania', 'Rwanda', 'Burundi']
            central_am = ['Guatemala', 'Honduras', 'Nicaragua', 'Costa Rica', 'El Salvador', 'Panama']
            if country in african:
                return 'East Africa'
            elif country in central_am:
                return 'Central America'
            return 'Other'

        df_reg = filtered_df.copy()
        df_reg['Region'] = df_reg['Country of Origin'].apply(classify_region)
        df_regional = df_reg[df_reg['Region'] != 'Other'].copy()

        if len(df_regional) > 0:
            defect_summary = df_regional.groupby('Region')[['Category One Defects', 'Category Two Defects']].mean().reset_index()
            defect_melted = defect_summary.melt(
                id_vars='Region', 
                var_name='Defect Type', 
                value_name='Average Defects'
            )
            defect_melted['Defect Type'] = defect_melted['Defect Type'].replace({
                'Category One Defects': 'Primary (Cat 1) Defects',
                'Category Two Defects': 'Secondary (Cat 2) Defects'
            })

            fig2 = px.bar(
                defect_melted,
                x='Defect Type',
                y='Average Defects',
                color='Region',
                barmode='group',
                color_discrete_map={'East Africa': '#8c6239', 'Central America': '#c49a6c'},
                labels={'Average Defects': 'Average Defects per Batch (350g)'}
            )
            fig2.update_layout(
                title="<b>Regional Cleanliness:</b> Central America vs. East Africa",
                legend=dict(title="Region", yanchor="top", y=0.95, xanchor="right", x=0.95),
                plot_bgcolor="white"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("ℹ️ Select at least one East African (e.g., Ethiopia) or Central American country (e.g., Guatemala) to compare regional defect profiles!")
            
    with col_right:
        # Visual 6: Processing methods in elite tier (Question 10)
        score_thresh = filtered_df['Total Cup Points'].quantile(0.80) if len(filtered_df) > 5 else 0
        df_elite = filtered_df[filtered_df['Total Cup Points'] >= score_thresh].copy()
        
        if len(df_elite) > 0:
            elite_processing = df_elite['Processing Method'].value_counts(normalize=True).reset_index()
            elite_processing.columns = ['Processing Method', 'Proportion']
            elite_processing['Percentage'] = elite_processing['Proportion'] * 100
            
            fig10 = px.bar(
                elite_processing,
                x='Percentage',
                y='Processing Method',
                orientation='h',
                color_discrete_sequence=['#8c6239'],
                labels={'Percentage': 'Proportion of Top 20% Coffees (%)'}
            )
            fig10.update_layout(
                title="<b>The Elite Tier:</b> Processing Methods of Top-Scoring Beans",
                plot_bgcolor="white"
            )
            st.plotly_chart(fig10, use_container_width=True)
        else:
            st.info("Insufficient data to analyze elite tier.")
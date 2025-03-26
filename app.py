import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.visualization import (
    create_premium_by_vehicle_type_chart,
    create_premium_by_year_chart,
    create_claim_distribution_chart,
    create_premium_insured_value_scatter,
    create_premium_by_make_chart,
    create_premium_distribution_chart,
    create_heatmap_correlation,
    create_claim_by_usage_chart,
    create_custom_dimension_chart,
    create_scatter_by_dimension,
    create_distribution_by_dimension,
    create_claims_histogram_by_dimension
)
from src.model import predict_premium, load_model_for_prediction

# Set page config
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    """Load the preprocessed insurance data"""
    return pd.read_csv('/Users/yuganthareshsoni/InsurancePremiumPredictor/data/insurance_cleaned_colab.csv')

# Custom CSS
def apply_custom_css():
    """Apply custom CSS for better styling"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #1E88E5 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        padding-top: 2rem !important;
    }
    .sub-header {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #0D47A1 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    .card {
        background-color: #f5f7ff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #0D47A1 !important;
    }
    .metric-label {
        font-size: 1rem !important;
        color: #666666 !important;
    }
    .prediction-result {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 20px;
        margin-top: 30px;
        text-align: center;
    }
    .prediction-value {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        color: #0D47A1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_landing_page():
    """Render the landing page"""
    st.markdown('<h1 class="main-header">Welcome to Insurance Premium Analysis and Calculator</h1>', unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("/Users/yuganthareshsoni/InsurancePremiumPredictor/landing_image.webp", use_column_width=True)
        
        st.markdown("""
        <div class="card">
        <h3 style="text-align: center; color: #1976D2;">Your comprehensive solution for insurance premium analysis and prediction</h3>
        <p style="text-align: center; font-size: 1.2rem;">
        Explore trends, analyze factors affecting premiums, and predict insurance costs using advanced analytics.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Click to Begin Analysis", use_container_width=True, key="begin_button"):
            st.session_state.page = "dashboard"
            st.rerun()

def render_dashboard(df):
    """Render the main dashboard"""
    st.markdown('<h1 class="main-header">Insurance Premium Dashboard</h1>', unsafe_allow_html=True)
    
    # Key metrics on top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="card">
        <p class="metric-label">Average Premium</p>
        <p class="metric-value">${df['PREMIUM'].mean():.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
        <p class="metric-label">Total Insured Value</p>
        <p class="metric-value">${df['INSURED_VALUE'].sum() / 1_000_000:.1f}M</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="card">
        <p class="metric-label">Claims Rate</p>
        <p class="metric-value">{df['HAS_CLAIM'].mean() * 100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="card">
        <p class="metric-label">Vehicle Count</p>
        <p class="metric-value">{df['OBJECT_ID'].nunique():,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Overview section
    st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)
    
    with st.expander("View Sample Data"):
        st.dataframe(df.head(10))
    
    # Summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3>Numerical Features</h3>', unsafe_allow_html=True)
        st.dataframe(df.describe().T)
    
    with col2:
        st.markdown('<h3>Categorical Features</h3>', unsafe_allow_html=True)
        categorical_cols = ['TYPE_VEHICLE', 'MAKE', 'USAGE', 'EFFECTIVE_YR']
        cat_summary = {}
        for col in categorical_cols:
            cat_summary[col] = df[col].value_counts().nlargest(5).to_dict()
        
        st.json(cat_summary)
    
    # Charts section
    st.markdown('<h2 class="sub-header">Key Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_premium_by_vehicle_type_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_claim_distribution_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_premium_by_year_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_premium_insured_value_scatter(df)
        st.plotly_chart(fig, use_container_width=True)

def render_yearly_analysis(df):
    """Render analysis by year"""
    st.markdown('<h1 class="main-header">Analysis by Year</h1>', unsafe_allow_html=True)
    
    # Year selector
    years = sorted(df['PROD_YEAR'].dropna().unique().astype(int))
    selected_year = st.selectbox("Select Production Year", years)
    
    year_df = df[df['PROD_YEAR'] == selected_year]
    
    # Key metrics for selected year
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="card">
        <p class="metric-label">Average Premium</p>
        <p class="metric-value">${year_df['PREMIUM'].mean():.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
        <p class="metric-label">Vehicle Count</p>
        <p class="metric-value">{len(year_df):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="card">
        <p class="metric-label">Claims Rate</p>
        <p class="metric-value">{year_df['HAS_CLAIM'].mean() * 100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="card">
        <p class="metric-label">Avg Claim Value</p>
        <p class="metric-value">${year_df['CLAIM_PAID'].mean():.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Year specific visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_premium_by_make_chart(year_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_premium_distribution_chart(year_df)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_heatmap_correlation(year_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_claim_by_usage_chart(year_df, selected_year)
        st.plotly_chart(fig, use_container_width=True)

def render_custom_view(df):
    """Render a customizable view based on user selection"""
    st.markdown('<h1 class="main-header">Custom Analysis View</h1>', unsafe_allow_html=True)
    
    # Selection options
    st.markdown('<h2 class="sub-header">Select Dimensions for Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dimension = st.selectbox(
            "Select Primary Dimension",
            options=['TYPE_VEHICLE', 'MAKE', 'USAGE', 'PROD_YEAR', 'SEATS_NUM']
        )
    
    with col2:
        metric = st.selectbox(
            "Select Metric to Analyze",
            options=['PREMIUM', 'INSURED_VALUE', 'CLAIM_PAID', 'PREMIUM_RATIO']
        )
    
    with col3:
        agg_func = st.selectbox(
            "Select Aggregation Function",
            options=['Mean', 'Median', 'Sum', 'Count', 'Min', 'Max']
        )
    
    # Generate visualization
    fig, title = create_custom_dimension_chart(df, dimension, metric, agg_func)
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    st.markdown('<h2 class="sub-header">Additional Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if metric != 'INSURED_VALUE':
            fig = create_scatter_by_dimension(df, dimension, metric)
        else:
            fig = create_distribution_by_dimension(df, dimension, metric)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if dimension in ['TYPE_VEHICLE', 'MAKE', 'USAGE']:
            claims_data = df.groupby(dimension)['HAS_CLAIM'].mean().reset_index()
            claims_data = claims_data.sort_values('HAS_CLAIM', ascending=False)
            
            if len(claims_data) > 15:
                claims_data = claims_data.head(15)
            
            fig = px.bar(
                claims_data,
                x=dimension,
                y='HAS_CLAIM',
                title=f'Claim Probability by {dimension}',
                color='HAS_CLAIM',
                color_continuous_scale='RdYlGn_r',
                labels={
                    dimension: dimension.replace('_', ' '),
                    'HAS_CLAIM': 'Claim Probability'
                }
            )
            
            fig.update_layout(
                xaxis_title=dimension.replace('_', ' '),
                yaxis_title='Claim Probability',
                plot_bgcolor='white',
                height=400
            )
        else:
            fig = create_claims_histogram_by_dimension(df, dimension)
        
        st.plotly_chart(fig, use_container_width=True)

def render_premium_predictor(df):
    """Render the premium prediction page"""
    st.markdown('<h1 class="main-header">Predict Insurance Premium</h1>', unsafe_allow_html=True)
    
    # Load model info to get feature importance
    try:
        model_info = load_model_for_prediction()
        feature_info = model_info['feature_info']
        
        # Display feature importance if available
        if 'feature_importance' in feature_info:
            st.markdown('<h2 class="sub-header">Factors Affecting Premium</h2>', unsafe_allow_html=True)
            
            # Get the top 5 features
            importance_data = pd.DataFrame(feature_info['feature_importance']).sort_values('importance', ascending=False).head(5)
            
            # Create bar chart for feature importance
            fig = px.bar(
                importance_data,
                x='importance',
                y='feature',
                orientation='h',
                title='Top Factors Affecting Premium',
                color='importance',
                color_continuous_scale='Viridis',
                labels={'importance': 'Importance Score', 'feature': 'Factor'}
            )
            
            fig.update_layout(
                xaxis_title='Importance Score',
                yaxis_title='Factor',
                height=400,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig)
    except Exception as e:
        st.warning(f"Could not load model information: {e}")
        feature_info = None
    
    # Input form for prediction
    st.markdown('<h2 class="sub-header">Enter Vehicle Details</h2>', unsafe_allow_html=True)
    
    # Create form with 3 columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Vehicle details
        vehicle_type = st.selectbox(
            "Vehicle Type",
            options=sorted(df['TYPE_VEHICLE'].unique())
        )
        
        make = st.selectbox(
            "Vehicle Make",
            options=sorted(df['MAKE'].unique())
        )
        
        usage = st.selectbox(
            "Vehicle Usage",
            options=sorted(df['USAGE'].unique())
        )
    
    with col2:
        # Technical specifications
        prod_year = st.number_input(
            "Production Year",
            min_value=int(df['PROD_YEAR'].min()),
            max_value=2025,
            value=2010
        )
        
        seats_num = st.number_input(
            "Number of Seats",
            min_value=int(df['SEATS_NUM'].min()),
            max_value=int(df['SEATS_NUM'].max()),
            value=4
        )
        
        carrying_capacity = st.number_input(
            "Carrying Capacity",
            min_value=float(df['CARRYING_CAPACITY'].min()),
            max_value=float(df['CARRYING_CAPACITY'].max()),
            value=5.0
        )
        
        ccm_ton = st.number_input(
            "Engine Capacity (CCM) / Tonnage",
            min_value=float(df['CCM_TON'].min()),
            max_value=float(df['CCM_TON'].max()),
            value=2000.0
        )
    
    with col3:
        # Insurance details
        insured_value = st.number_input(
            "Insured Value ($)",
            min_value=float(df['INSURED_VALUE'].min()),
            max_value=float(df['INSURED_VALUE'].max()),
            value=300000.0,
            step=10000.0
        )
        
        sex = st.radio(
            "Owner Gender (0=Male, 1=Female)",
            options=[0, 1],
            horizontal=True
        )
        
        has_claim = st.radio(
            "Previous Claims (0=No, 1=Yes)",
            options=[0, 1],
            horizontal=True
        )
    
    # Prediction button
    if st.button("Calculate Premium", type="primary", use_container_width=True):
        # Create input data for prediction
        input_data = {
            'SEX': sex,
            'INSURED_VALUE': insured_value,
            'PROD_YEAR': prod_year,
            'SEATS_NUM': seats_num,
            'CARRYING_CAPACITY': carrying_capacity,
            'TYPE_VEHICLE': vehicle_type,
            'CCM_TON': ccm_ton,
            'MAKE': make,
            'USAGE': usage,
            'HAS_CLAIM': has_claim
        }
        
        try:
            # Make prediction
            with st.spinner('Calculating premium...'):
                premium_prediction = predict_premium(input_data)
            
            # Display prediction
            st.markdown("""
            <div class="prediction-result">
                <h3>Estimated Premium</h3>
                <p class="prediction-value">$%.2f</p>
            </div>
            """ % premium_prediction, unsafe_allow_html=True)
            
            # Find similar vehicles in the dataset
            similar_condition = (
                (df['TYPE_VEHICLE'] == vehicle_type) &
                (df['MAKE'] == make) &
                (df['USAGE'] == usage)
            )
            
            similar_vehicles = df[similar_condition].head(5)[
                ['PROD_YEAR', 'INSURED_VALUE', 'PREMIUM', 'HAS_CLAIM']
            ]
            
            if not similar_vehicles.empty:
                st.markdown('<h3>Similar Vehicles in Database</h3>', unsafe_allow_html=True)
                st.dataframe(similar_vehicles)
            
            # Recommendation based on prediction
            st.markdown("""
            <div class="card">
                <h4>Analysis</h4>
                <p>This premium estimate is based on advanced machine learning algorithms
                trained on thousands of insurance records. Factors like vehicle type, make,
                usage, and insured value significantly influence the premium calculation.</p>
                <p>For more accurate quotes, we recommend contacting an insurance agent.</p>
            </div>
            """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error calculating premium: {e}")
            st.info("Please check your inputs and try again.")

def main():
    """Main function to run the streamlit app"""
    apply_custom_css()
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "landing"
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Insurance Premium Predictor")
        st.image("/Users/yuganthareshsoni/InsurancePremiumPredictor/ferrari.webp", use_column_width=True)
        
        st.markdown("---")
        
        if st.button("Home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()
        
        if st.button("Dashboard Overview", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
        
        if st.button("Analysis by Year", use_container_width=True):
            st.session_state.page = "yearly"
            st.rerun()
        
        if st.button("Custom Analysis", use_container_width=True):
            st.session_state.page = "custom"
            st.rerun()
        
        if st.button("Predict Premium", use_container_width=True):
            st.session_state.page = "predictor"
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard provides comprehensive analysis of vehicle insurance data, 
        allowing users to explore trends and predict premiums based on various factors.
        """)
    
    # Conditional rendering based on the current page
    if st.session_state.page == "landing":
        render_landing_page()
    else:
        # Load data for all other pages
        df = load_data()
        
        if st.session_state.page == "dashboard":
            render_dashboard(df)
        elif st.session_state.page == "yearly":
            render_yearly_analysis(df)
        elif st.session_state.page == "custom":
            render_custom_view(df)
        elif st.session_state.page == "predictor":
            render_premium_predictor(df)

if __name__ == "__main__":
    main()
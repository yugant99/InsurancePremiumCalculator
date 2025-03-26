import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_premium_by_vehicle_type_chart(df):
    """
    Create a bar chart showing average premium by vehicle type
    
    Args:
        df: Input DataFrame
    
    Returns:
        Plotly figure object
    """
    vehicle_premium = df.groupby('TYPE_VEHICLE')['PREMIUM'].mean().reset_index()
    vehicle_premium = vehicle_premium.sort_values('PREMIUM', ascending=False).head(10)
    
    fig = px.bar(
        vehicle_premium,
        x='TYPE_VEHICLE',
        y='PREMIUM',
        title='Average Premium by Vehicle Type',
        color='PREMIUM',
        color_continuous_scale='Blues',
        labels={'TYPE_VEHICLE': 'Vehicle Type', 'PREMIUM': 'Average Premium ($)'}
    )
    
    fig.update_layout(
        xaxis_title='Vehicle Type',
        yaxis_title='Average Premium ($)',
        plot_bgcolor='white',
        height=400
    )
    
    return fig

def create_premium_by_year_chart(df):
    """
    Create a line chart showing premium trends by year
    
    Args:
        df: Input DataFrame
    
    Returns:
        Plotly figure object
    """
    df['PROD_YEAR'] = pd.to_numeric(df['PROD_YEAR'], errors='coerce')
    year_premium = df.groupby('PROD_YEAR')['PREMIUM'].mean().reset_index()
    year_premium = year_premium.sort_values('PROD_YEAR')
    
    # Filter out years with too few data points
    year_counts = df['PROD_YEAR'].value_counts().reset_index()
    year_counts.columns = ['PROD_YEAR', 'count']
    year_premium = year_premium.merge(year_counts, on='PROD_YEAR')
    year_premium = year_premium[year_premium['count'] > 10]
    
    fig = px.line(
        year_premium,
        x='PROD_YEAR',
        y='PREMIUM',
        title='Average Premium by Vehicle Production Year',
        markers=True,
        labels={'PROD_YEAR': 'Production Year', 'PREMIUM': 'Average Premium ($)'}
    )
    
    fig.update_layout(
        xaxis_title='Production Year',
        yaxis_title='Average Premium ($)',
        plot_bgcolor='white',
        height=400
    )
    
    return fig

def create_claim_distribution_chart(df):
    """
    Create a pie chart showing claim distribution
    
    Args:
        df: Input DataFrame
    
    Returns:
        Plotly figure object
    """
    claim_counts = df['HAS_CLAIM'].value_counts().reset_index()
    claim_counts.columns = ['HAS_CLAIM', 'Count']
    claim_counts['HAS_CLAIM'] = claim_counts['HAS_CLAIM'].map({0: 'No Claim', 1: 'Has Claim'})
    
    fig = px.pie(
        claim_counts,
        values='Count',
        names='HAS_CLAIM',
        title='Claims Distribution',
        color_discrete_sequence=px.colors.sequential.Blues_r,
        hole=0.4
    )
    
    fig.update_layout(
        legend_title='Claim Status',
        plot_bgcolor='white',
        height=400
    )
    
    return fig

def create_premium_insured_value_scatter(df):
    """
    Create a scatter plot of premium vs insured value
    
    Args:
        df: Input DataFrame
    
    Returns:
        Plotly figure object
    """
    # Sample to avoid overcrowding
    sample_df = df.sample(min(5000, len(df)))
    
    fig = px.scatter(
        sample_df,
        x='INSURED_VALUE',
        y='PREMIUM',
        color='HAS_CLAIM',
        title='Premium vs Insured Value',
        color_discrete_sequence=['#00CC96', '#EF553B'],
        opacity=0.7,
        labels={
            'INSURED_VALUE': 'Insured Value ($)',
            'PREMIUM': 'Premium ($)',
            'HAS_CLAIM': 'Has Claim'
        }
    )
    
    fig.update_layout(
        xaxis_title='Insured Value ($)',
        yaxis_title='Premium ($)',
        plot_bgcolor='white',
        height=400
    )
    
    return fig

def create_premium_by_make_chart(df):
    """
    Create a bar chart showing premium by vehicle make
    
    Args:
        df: Input DataFrame
    
    Returns:
        Plotly figure object
    """
    make_premium = df.groupby('MAKE')['PREMIUM'].mean().reset_index()
    make_premium = make_premium.sort_values('PREMIUM', ascending=False).head(10)
    
    fig = px.bar(
        make_premium,
        x='MAKE',
        y='PREMIUM',
        title='Average Premium by Vehicle Make',
        color='PREMIUM',
        color_continuous_scale='Viridis',
        labels={'MAKE': 'Vehicle Make', 'PREMIUM': 'Average Premium ($)'}
    )
    
    fig.update_layout(
        xaxis_title='Vehicle Make',
        yaxis_title='Average Premium ($)',
        plot_bgcolor='white',
        height=400
    )
    
    return fig

def create_premium_distribution_chart(df):
    """
    Create a histogram of premium distribution
    
    Args:
        df: Input DataFrame
    
    Returns:
        Plotly figure object
    """
    fig = px.histogram(
        df,
        x='PREMIUM',
        nbins=50,
        title='Premium Distribution',
        color_discrete_sequence=['#1E88E5'],
        labels={'PREMIUM': 'Premium ($)'}
    )
    
    fig.update_layout(
        xaxis_title='Premium ($)',
        yaxis_title='Count',
        plot_bgcolor='white',
        height=400
    )
    
    return fig

def create_heatmap_correlation(df):
    """
    Create a correlation heatmap for numerical features
    
    Args:
        df: Input DataFrame
    
    Returns:
        Plotly figure object
    """
    numeric_cols = ['PREMIUM', 'INSURED_VALUE', 'SEATS_NUM', 'CARRYING_CAPACITY', 'CCM_TON', 'PREMIUM_RATIO']
    corr_df = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_df,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Heatmap'
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_claim_by_usage_chart(df, year=None):
    """
    Create a bar chart showing claim probability by usage
    
    Args:
        df: Input DataFrame
        year: Optional year filter
    
    Returns:
        Plotly figure object
    """
    title = 'Claim Probability by Usage'
    if year:
        title += f' for {year}'
    
    year_claims_df = df.groupby('USAGE')['HAS_CLAIM'].mean().reset_index()
    fig = px.bar(
        year_claims_df,
        x='USAGE',
        y='HAS_CLAIM',
        title=title,
        color='HAS_CLAIM',
        color_continuous_scale='Viridis',
        labels={'USAGE': 'Vehicle Usage', 'HAS_CLAIM': 'Claim Probability'}
    )
    
    fig.update_layout(
        xaxis_title='Vehicle Usage',
        yaxis_title='Claim Probability',
        plot_bgcolor='white',
        height=400
    )
    
    return fig

def create_custom_dimension_chart(df, dimension, metric, agg_func):
    """
    Create a custom bar chart based on user-selected dimensions and metrics
    
    Args:
        df: Input DataFrame
        dimension: Column to group by
        metric: Column to aggregate
        agg_func: Aggregation function name
    
    Returns:
        Plotly figure object and title
    """
    # Map aggregation function
    agg_map = {
        'Mean': 'mean',
        'Median': 'median',
        'Sum': 'sum',
        'Count': 'count',
        'Min': 'min',
        'Max': 'max'
    }
    
    # Generate the aggregated data
    if dimension == 'PROD_YEAR':
        df['PROD_YEAR'] = pd.to_numeric(df['PROD_YEAR'], errors='coerce')
    
    agg_data = df.groupby(dimension)[metric].agg(agg_map[agg_func]).reset_index()
    agg_data = agg_data.sort_values(metric, ascending=False)
    
    # Filter to top values if needed
    if len(agg_data) > 15:
        agg_data = agg_data.head(15)
    
    # Generate title
    title = f'{agg_func} {metric} by {dimension}'
    
    fig = px.bar(
        agg_data,
        x=dimension,
        y=metric,
        title=title,
        color=metric,
        color_continuous_scale='Viridis',
        labels={dimension: dimension.replace('_', ' '), metric: metric.replace('_', ' ')}
    )
    
    fig.update_layout(
        xaxis_title=dimension.replace('_', ' '),
        yaxis_title=metric.replace('_', ' '),
        plot_bgcolor='white',
        height=500
    )
    
    return fig, title

def create_scatter_by_dimension(df, dimension, metric):
    """
    Create a scatter plot of metric vs insured value, colored by dimension
    
    Args:
        df: Input DataFrame
        dimension: Column to use for color
        metric: Y-axis metric
    
    Returns:
        Plotly figure object
    """
    sample_df = df.sample(min(3000, len(df)))
    
    fig = px.scatter(
        sample_df,
        x='INSURED_VALUE',
        y=metric,
        color=dimension,
        title=f'{metric} vs Insured Value by {dimension}',
        labels={
            'INSURED_VALUE': 'Insured Value ($)',
            metric: metric.replace('_', ' '),
            dimension: dimension.replace('_', ' ')
        }
    )
    
    fig.update_layout(
        xaxis_title='Insured Value ($)',
        yaxis_title=metric.replace('_', ' '),
        plot_bgcolor='white',
        height=400
    )
    
    return fig

def create_distribution_by_dimension(df, dimension, metric):
    """
    Create a box plot showing distribution of metric by dimension
    
    Args:
        df: Input DataFrame
        dimension: X-axis grouping
        metric: Y-axis metric
    
    Returns:
        Plotly figure object
    """
    fig = px.box(
        df,
        x=dimension,
        y=metric,
        title=f'Distribution of {metric} by {dimension}',
        color=dimension,
        labels={
            dimension: dimension.replace('_', ' '),
            metric: metric.replace('_', ' ')
        }
    )
    
    fig.update_layout(
        xaxis_title=dimension.replace('_', ' '),
        yaxis_title=metric.replace('_', ' '),
        plot_bgcolor='white',
        height=400,
        showlegend=False
    )
    
    return fig

def create_claims_histogram_by_dimension(df, dimension):
    """
    Create a histogram showing claims distribution by a numerical dimension
    
    Args:
        df: Input DataFrame
        dimension: X-axis dimension
    
    Returns:
        Plotly figure object
    """
    fig = px.histogram(
        df,
        x=dimension,
        color='HAS_CLAIM',
        barmode='group',
        title=f'Claims Distribution by {dimension}',
        labels={
            dimension: dimension.replace('_', ' '),
            'HAS_CLAIM': 'Has Claim'
        },
        color_discrete_map={0: '#00CC96', 1: '#EF553B'}
    )
    
    fig.update_layout(
        xaxis_title=dimension.replace('_', ' '),
        yaxis_title='Count',
        plot_bgcolor='white',
        height=400
    )
    
    return fig
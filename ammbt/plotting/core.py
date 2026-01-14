"""
Core plotting functions using Plotly.

Interactive visualizations for backtest results.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


def plot_performance(
    result,
    strategy_idx: int = 0,
    show_reserves: bool = True,
) -> go.Figure:
    """
    Plot performance for a specific strategy.

    Parameters
    ----------
    result : BacktestResult
        Backtest result object
    strategy_idx : int
        Which strategy to plot
    show_reserves : bool
        Whether to show pool reserve history

    Returns
    -------
    go.Figure
        Plotly figure
    """
    position_history = result.get_position_history(strategy_idx)

    # Calculate total value over time
    if 'price' in result.metadata:
        prices = result.metadata['price']
    else:
        prices = result.metadata['reserve1_history'] / result.metadata['reserve0_history']

    values = (
        position_history['token0_balance'] * prices +
        position_history['token1_balance'] +
        position_history['uncollected_fees_0'] * prices +
        position_history['uncollected_fees_1']
    )

    # Create subplots
    n_rows = 3 if show_reserves else 2
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=(
            'Portfolio Value Over Time',
            'Uncollected Fees',
            'Pool Reserves' if show_reserves else None,
        ),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3] if show_reserves else [0.5, 0.5],
    )

    # Plot 1: Portfolio value
    fig.add_trace(
        go.Scatter(
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00D9FF', width=2),
        ),
        row=1, col=1,
    )

    # Plot 2: Fees
    fig.add_trace(
        go.Scatter(
            y=position_history['uncollected_fees_0'],
            mode='lines',
            name='Fees Token0',
            line=dict(color='#FF6B6B', width=1.5),
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=position_history['uncollected_fees_1'],
            mode='lines',
            name='Fees Token1',
            line=dict(color='#4ECDC4', width=1.5),
        ),
        row=2, col=1,
    )

    # Plot 3: Reserves (if enabled)
    if show_reserves and 'reserve0_history' in result.metadata:
        fig.add_trace(
            go.Scatter(
                y=result.metadata['reserve0_history'],
                mode='lines',
                name='Reserve Token0',
                line=dict(color='#FF6B6B', width=1.5),
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=result.metadata['reserve1_history'],
                mode='lines',
                name='Reserve Token1',
                line=dict(color='#4ECDC4', width=1.5),
            ),
            row=3, col=1,
        )

    # Update layout
    fig.update_layout(
        title=f'LP Performance - Strategy {strategy_idx}',
        height=800 if show_reserves else 600,
        showlegend=True,
        template='plotly_dark',
        hovermode='x unified',
    )

    fig.update_xaxes(title_text='Swap Index', row=n_rows, col=1)
    fig.update_yaxes(title_text='Value (USD)', row=1, col=1)
    fig.update_yaxes(title_text='Fees', row=2, col=1)
    if show_reserves:
        fig.update_yaxes(title_text='Reserves', row=3, col=1)

    return fig


def plot_metrics_heatmap(
    result,
    x_param: str = 'rebalance_threshold',
    y_param: str = 'initial_capital',
    metric: str = 'net_pnl',
) -> go.Figure:
    """
    Plot heatmap of metrics across strategy parameter space.

    Parameters
    ----------
    result : BacktestResult
        Backtest result
    x_param : str
        Parameter for x-axis
    y_param : str
        Parameter for y-axis
    metric : str
        Metric to plot

    Returns
    -------
    go.Figure
        Plotly heatmap
    """
    # Combine strategy params with metrics
    data = pd.concat([result.strategy_params, result.metrics], axis=1)

    # Pivot for heatmap
    heatmap_data = data.pivot_table(
        values=metric,
        index=y_param,
        columns=x_param,
        aggfunc='mean',
    )

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn',
        colorbar=dict(title=metric),
        hovertemplate=(
            f'{x_param}: %{{x}}<br>'
            f'{y_param}: %{{y}}<br>'
            f'{metric}: %{{z:.2f}}<br>'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        title=f'{metric} across {x_param} and {y_param}',
        xaxis_title=x_param,
        yaxis_title=y_param,
        template='plotly_dark',
        height=600,
    )

    return fig


def plot_efficient_frontier(
    result,
    risk_metric: str = 'max_drawdown_pct',
    return_metric: str = 'total_return_pct',
) -> go.Figure:
    """
    Plot efficient frontier (risk vs return).

    Parameters
    ----------
    result : BacktestResult
        Backtest result
    risk_metric : str
        Risk metric for x-axis
    return_metric : str
        Return metric for y-axis

    Returns
    -------
    go.Figure
        Plotly scatter plot
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.metrics[risk_metric],
        y=result.metrics[return_metric],
        mode='markers',
        marker=dict(
            size=8,
            color=result.metrics['sharpe'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Sharpe Ratio'),
        ),
        text=[f'Strategy {i}' for i in range(len(result.metrics))],
        hovertemplate=(
            '<b>%{text}</b><br>'
            f'{risk_metric}: %{{x:.2f}}<br>'
            f'{return_metric}: %{{y:.2f}}<br>'
            'Sharpe: %{marker.color:.2f}<br>'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title=risk_metric,
        yaxis_title=return_metric,
        template='plotly_dark',
        height=600,
        width=800,
    )

    return fig


def plot_pnl_distribution(result) -> go.Figure:
    """
    Plot distribution of PnL across all strategies.

    Parameters
    ----------
    result : BacktestResult
        Backtest result

    Returns
    -------
    go.Figure
        Plotly histogram
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=result.metrics['net_pnl'],
        nbinsx=30,
        marker=dict(
            color='#00D9FF',
            line=dict(color='white', width=1),
        ),
        name='Net PnL',
    ))

    # Add mean line
    mean_pnl = result.metrics['net_pnl'].mean()
    fig.add_vline(
        x=mean_pnl,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Mean: {mean_pnl:.2f}',
    )

    fig.update_layout(
        title='PnL Distribution Across Strategies',
        xaxis_title='Net PnL (USD)',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=500,
        width=800,
        showlegend=False,
    )

    return fig

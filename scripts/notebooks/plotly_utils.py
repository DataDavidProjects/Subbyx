from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px


PURPLE_THEME = {
    "primary": "#7C3AED",
    "primary_light": "#A78BFA",
    "primary_dark": "#5B21B6",
    "secondary": "#8B5CF6",
    "background": "#FAFAFA",
    "paper": "#FFFFFF",
    "text": "#1F2937",
    "text_light": "#6B7280",
    "grid": "#E5E7EB",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "colorscale": [
        "#EDE9FE",
        "#DDD6FE",
        "#C4B5FD",
        "#A78BFA",
        "#8B5CF6",
        "#7C3AED",
        "#6D28D9",
        "#5B21B6",
        "#4C1D95",
        "#3B0764",
    ],
}

px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = PURPLE_THEME["colorscale"]


def configure_layout(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        font=dict(family="Inter, sans-serif", size=12, color=PURPLE_THEME["text"]),
        plot_bgcolor=PURPLE_THEME["paper"],
        paper_bgcolor=PURPLE_THEME["paper"],
        title_font=dict(size=16, color=PURPLE_THEME["primary_dark"]),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def add_fraud_threshold_line(
    fig: go.Figure, x: list, threshold: float = 15, y_max: float | None = None
) -> go.Figure:
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=PURPLE_THEME["error"],
        line_width=2,
        annotation_text=f"Fraud threshold ({threshold}d)",
        annotation_position="top right",
    )
    return fig


def create_histogram(
    df: pd.DataFrame,
    column: str,
    title: str | None = None,
    bins: int = 30,
    color: str | None = None,
    add_fraud_line: bool = False,
) -> go.Figure:
    if title is None:
        title = f"Distribution of {column}"
    if color is None:
        color = PURPLE_THEME["primary"]

    fig = px.histogram(df, x=column, nbins=bins, title=title, color_discrete_sequence=[color])
    fig.update_traces(marker_line_color="white", marker_line_width=0.5, opacity=0.8)

    if add_fraud_line and column == "dunning_days":
        add_fraud_threshold_line(fig, df[column].dropna())

    configure_layout(fig, title)
    return fig


def create_box_plot(
    df: pd.DataFrame,
    column: str,
    title: str | None = None,
    color: str | None = None,
) -> go.Figure:
    if title is None:
        title = f"Box Plot of {column}"
    if color is None:
        color = PURPLE_THEME["primary"]

    fig = px.box(df, y=column, title=title, color_discrete_sequence=[color])
    configure_layout(fig, title)
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    orientation: str = "v",
    color: str | None = None,
    top_n: int | None = None,
) -> go.Figure:
    if color is None:
        color = PURPLE_THEME["primary"]

    if top_n is not None:
        df = df.head(top_n)

    if orientation == "h":
        fig = px.bar(df, y=x, x=y, title=title, orientation="h", color_discrete_sequence=[color])
    else:
        fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=[color])

    configure_layout(fig, title)
    return fig


def create_horizontal_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
) -> go.Figure:
    return create_bar_chart(df, x, y, title, orientation="h", color=color)


def create_pie_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    title: str,
    hole: bool = False,
) -> go.Figure:
    if hole:
        fig = px.pie(df, names=names, values=values, title=title, hole=0.4)
    else:
        fig = px.pie(df, names=names, values=values, title=title)

    fig.update_traces(textposition="inside", textinfo="percent+label")
    configure_layout(fig, title, height=350)
    return fig


def create_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str | list,
    title: str,
    markers: bool = True,
    color: str | None = None,
) -> go.Figure:
    if color is None:
        color = PURPLE_THEME["primary"]

    if isinstance(y, list):
        fig = go.Figure()
        for col in y:
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[col],
                    mode="lines+markers" if markers else "lines",
                    name=col,
                    line=dict(color=color),
                )
            )
    else:
        fig = px.line(df, x=x, y=y, title=title, markers=markers)
        fig.update_traces(line_color=color)

    configure_layout(fig, title)
    return fig


def create_funnel(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
) -> go.Figure:
    if color is None:
        color = PURPLE_THEME["primary"]

    fig = px.funnel(df, x=x, y=y, title=title, color_discrete_sequence=[color])
    configure_layout(fig, title)
    return fig


def create_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color_col: str | None = None,
    size_col: str | None = None,
) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, title=title, color=color_col, size=size_col)
    configure_layout(fig, title)
    return fig


def create_stacked_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, title=title, barmode="stack")
    configure_layout(fig, title)
    return fig


def create_gauge(
    value: float,
    title: str,
    max_value: float = 100,
    threshold: float | None = None,
) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [None, max_value]},
                "bar": {"color": PURPLE_THEME["primary"]},
                "bgcolor": PURPLE_THEME["paper"],
                "borderwidth": 2,
                "bordercolor": PURPLE_THEME["grid"],
                "steps": [
                    {"range": [0, max_value * 0.6], "color": PURPLE_THEME["colorscale"][2]},
                    {
                        "range": [max_value * 0.6, max_value * 0.8],
                        "color": PURPLE_THEME["colorscale"][4],
                    },
                    {"range": [max_value * 0.8, max_value], "color": PURPLE_THEME["colorscale"][6]},
                ],
            },
        )
    )

    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color=PURPLE_THEME["error"])

    configure_layout(fig, title, height=300)
    return fig


def create_treemap(
    df: pd.DataFrame,
    path: str | list,
    values: str,
    title: str,
) -> go.Figure:
    fig = px.treemap(df, path=path, values=values, title=title)
    configure_layout(fig, title, height=400)
    return fig


def save_html(fig: go.Figure, filename: str, plots_dir: Path) -> None:
    filepath = plots_dir / f"{filename}.png"
    fig.write_image(filepath, scale=2)


def save_static(fig: go.Figure, filename: str, plots_dir: Path, format: str = "png") -> None:
    filepath = plots_dir / f"{filename}.{format}"
    fig.write_image(filepath, scale=2)


import pandas as pd
from pathlib import Path

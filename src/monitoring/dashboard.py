#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for MODIS Land Cover Classification MLOps Pipeline.
Provides comprehensive monitoring of model performance, data drift, and system health.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, callback, dcc, html
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLOpsDashboard:
    """
    Real-time monitoring dashboard for MLOps pipeline.

    Features:
    - Model performance monitoring
    - Data drift detection
    - System health metrics
    - Prediction volume tracking
    - Alert management
    - Historical trend analysis
    """

    def __init__(
        self,
        model_server_url: str = "http://localhost:5001",
        monitoring_db_path: str = "monitoring.db",
        refresh_interval: int = 30,  # seconds
    ):
        self.model_server_url = model_server_url
        self.monitoring_db_path = monitoring_db_path
        self.refresh_interval = refresh_interval

        # Initialize database
        self._init_database()

        # Initialize Dash app
        self.app = dash.Dash(__name__, title="MODIS MLOps Monitoring Dashboard")
        self._setup_layout()
        self._setup_callbacks()

        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self.monitoring_thread.start()

    def _init_database(self):
        """Initialize SQLite database for monitoring data."""
        conn = sqlite3.connect(self.monitoring_db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                server_status TEXT,
                response_time REAL,
                memory_usage REAL,
                cpu_usage REAL,
                model_loaded BOOLEAN
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                prediction_value INTEGER,
                confidence REAL,
                processing_time REAL,
                input_features TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                drift_detected BOOLEAN,
                drift_score REAL,
                features_with_drift INTEGER,
                data_quality_score REAL,
                drift_method TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        """
        )

        conn.commit()
        conn.close()

    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div(
            [
                # Header
                html.Div(
                    [
                        html.H1(
                            "🛰️ MODIS Land Cover MLOps Monitoring Dashboard",
                            className="dashboard-title",
                        ),
                        html.P(
                            f"Real-time monitoring of model performance and system health",
                            className="dashboard-subtitle",
                        ),
                        html.Div(id="last-update", className="last-update"),
                    ],
                    className="header",
                ),
                # Alert Bar
                html.Div(id="alert-bar", className="alert-bar"),
                # Main Content
                html.Div(
                    [
                        # Top Row - Key Metrics
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("System Health"),
                                        html.Div(
                                            id="system-health-indicator", className="metric-card"
                                        ),
                                    ],
                                    className="metric-box",
                                ),
                                html.Div(
                                    [
                                        html.H3("Model Status"),
                                        html.Div(
                                            id="model-status-indicator", className="metric-card"
                                        ),
                                    ],
                                    className="metric-box",
                                ),
                                html.Div(
                                    [
                                        html.H3("Predictions/Hour"),
                                        html.Div(id="prediction-rate", className="metric-card"),
                                    ],
                                    className="metric-box",
                                ),
                                html.Div(
                                    [
                                        html.H3("Data Drift Status"),
                                        html.Div(id="drift-status", className="metric-card"),
                                    ],
                                    className="metric-box",
                                ),
                            ],
                            className="metrics-row",
                        ),
                        # Second Row - Charts
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(id="response-time-chart"),
                                    ],
                                    className="chart-container",
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(id="prediction-distribution"),
                                    ],
                                    className="chart-container",
                                ),
                            ],
                            className="charts-row",
                        ),
                        # Third Row - Detailed Monitoring
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(id="drift-monitoring-chart"),
                                    ],
                                    className="chart-container",
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(id="model-performance-trend"),
                                    ],
                                    className="chart-container",
                                ),
                            ],
                            className="charts-row",
                        ),
                        # Fourth Row - Recent Activity
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("Recent Predictions"),
                                        html.Div(id="recent-predictions-table"),
                                    ],
                                    className="table-container",
                                ),
                                html.Div(
                                    [
                                        html.H3("Active Alerts"),
                                        html.Div(id="alerts-table"),
                                    ],
                                    className="table-container",
                                ),
                            ],
                            className="tables-row",
                        ),
                    ],
                    className="main-content",
                ),
                # Auto-refresh component
                dcc.Interval(
                    id="interval-component",
                    interval=self.refresh_interval * 1000,  # in milliseconds
                    n_intervals=0,
                ),
                # Store components for data
                dcc.Store(id="system-data"),
                dcc.Store(id="prediction-data"),
                dcc.Store(id="drift-data"),
                dcc.Store(id="alert-data"),
            ],
            className="dashboard-container",
        )

    def _setup_callbacks(self):  # noqa: C901
        """Setup Dash callbacks."""
        self._setup_data_callbacks()
        self._setup_indicator_callbacks()
        self._setup_chart_callbacks()

    def _setup_data_callbacks(self):
        """Setup data update callbacks."""

        @self.app.callback(
            [
                Output("system-data", "data"),
                Output("prediction-data", "data"),
                Output("drift-data", "data"),
                Output("alert-data", "data"),
                Output("last-update", "children"),
            ],
            [Input("interval-component", "n_intervals")],
        )
        def update_data(n):
            """Update all data stores."""
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Get data from database
            system_data = self._get_system_health_data()
            prediction_data = self._get_prediction_data()
            drift_data = self._get_drift_data()
            alert_data = self._get_alert_data()

            return (
                system_data,
                prediction_data,
                drift_data,
                alert_data,
                f"Last updated: {current_time}",
            )

    def _setup_indicator_callbacks(self):
        """Setup indicator update callbacks."""

        @self.app.callback(
            [
                Output("system-health-indicator", "children"),
                Output("model-status-indicator", "children"),
                Output("prediction-rate", "children"),
                Output("drift-status", "children"),
            ],
            [
                Input("system-data", "data"),
                Input("prediction-data", "data"),
                Input("drift-data", "data"),
            ],
        )
        def update_indicators(system_data, prediction_data, drift_data):
            """Update key indicator cards."""

            # System Health
            if system_data and len(system_data) > 0:
                latest_health = system_data[-1]
                health_status = (
                    "🟢 Healthy" if latest_health["server_status"] == "healthy" else "🔴 Unhealthy"
                )
                response_time = f"{latest_health['response_time']:.2f}s"
                health_indicator = html.Div(
                    [
                        html.Div(health_status, className="status-text"),
                        html.Div(f"Response: {response_time}", className="detail-text"),
                    ]
                )
            else:
                health_indicator = html.Div("🔴 No Data", className="status-text")

            # Model Status
            if system_data and len(system_data) > 0:
                latest_health = system_data[-1]
                model_status = "🟢 Loaded" if latest_health["model_loaded"] else "🔴 Not Loaded"
                model_indicator = html.Div(model_status, className="status-text")
            else:
                model_indicator = html.Div("🔴 Unknown", className="status-text")

            # Prediction Rate
            if prediction_data and len(prediction_data) > 0:
                # Calculate predictions in last hour
                one_hour_ago = datetime.now() - timedelta(hours=1)
                recent_predictions = [
                    p
                    for p in prediction_data
                    if datetime.fromisoformat(p["timestamp"]) > one_hour_ago
                ]
                rate = len(recent_predictions)
                rate_indicator = html.Div(
                    [
                        html.Div(str(rate), className="metric-number"),
                        html.Div("predictions", className="metric-unit"),
                    ]
                )
            else:
                rate_indicator = html.Div("0", className="metric-number")

            # Drift Status
            if drift_data and len(drift_data) > 0:
                latest_drift = drift_data[-1]
                drift_detected = latest_drift["drift_detected"]
                drift_score = latest_drift["drift_score"]

                if drift_detected:
                    drift_status = f"🔴 Drift Detected ({drift_score:.3f})"
                else:
                    drift_status = f"🟢 No Drift ({drift_score:.3f})"

                drift_indicator = html.Div(drift_status, className="status-text")
            else:
                drift_indicator = html.Div("🔴 No Data", className="status-text")

            return health_indicator, model_indicator, rate_indicator, drift_indicator

    def _setup_chart_callbacks(self):  # noqa: C901
        """Setup chart update callbacks."""

        @self.app.callback(Output("response-time-chart", "figure"), [Input("system-data", "data")])
        def update_response_time_chart(system_data):
            """Update response time chart."""
            if not system_data:
                return go.Figure().add_annotation(text="No data available")

            df = pd.DataFrame(system_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["response_time"],
                    mode="lines+markers",
                    name="Response Time",
                    line=dict(color="#1f77b4"),
                )
            )

            fig.update_layout(
                title="Response Time Trend",
                xaxis_title="Time",
                yaxis_title="Response Time (seconds)",
                height=300,
            )

            return fig

        @self.app.callback(
            Output("prediction-distribution", "figure"), [Input("prediction-data", "data")]
        )
        def update_prediction_distribution(prediction_data):
            """Update prediction distribution chart."""
            if not prediction_data:
                return go.Figure().add_annotation(text="No data available")

            df = pd.DataFrame(prediction_data)

            # Count predictions by value
            prediction_counts = df["prediction_value"].value_counts().sort_index()

            # Land cover class names (simplified)
            class_names = {
                1: "Evergreen Forest",
                2: "Deciduous Forest",
                3: "Mixed Forest",
                4: "Closed Shrublands",
                5: "Open Shrublands",
                8: "Woody Savannas",
                10: "Grasslands",
            }

            labels = [
                f"{idx}: {class_names.get(idx, f'Class {idx}')}" for idx in prediction_counts.index
            ]

            fig = go.Figure(data=[go.Pie(labels=labels, values=prediction_counts.values, hole=0.3)])

            fig.update_layout(title="Prediction Distribution (Land Cover Classes)", height=300)

            return fig

        @self.app.callback(
            Output("drift-monitoring-chart", "figure"), [Input("drift-data", "data")]
        )
        def update_drift_chart(drift_data):
            """Update drift monitoring chart."""
            if not drift_data:
                return go.Figure().add_annotation(text="No data available")

            df = pd.DataFrame(drift_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Create subplot for drift score and data quality
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Drift Score", "Data Quality Score"),
                vertical_spacing=0.1,
            )

            # Drift score
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["drift_score"],
                    mode="lines+markers",
                    name="Drift Score",
                    line=dict(color="red"),
                ),
                row=1,
                col=1,
            )

            # Add threshold line
            fig.add_hline(
                y=0.1,
                line_dash="dash",
                line_color="orange",
                annotation_text="Drift Threshold",
                row=1,
                col=1,
            )

            # Data quality score
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["data_quality_score"],
                    mode="lines+markers",
                    name="Data Quality",
                    line=dict(color="green"),
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                title="Data Drift and Quality Monitoring", height=400, showlegend=False
            )

            return fig

        @self.app.callback(
            Output("recent-predictions-table", "children"), [Input("prediction-data", "data")]
        )
        def update_predictions_table(prediction_data):
            """Update recent predictions table."""
            if not prediction_data:
                return html.Div("No prediction data available")

            # Get last 10 predictions
            recent = prediction_data[-10:]

            table_rows = []
            for pred in reversed(recent):  # Most recent first
                timestamp = datetime.fromisoformat(pred["timestamp"]).strftime("%H:%M:%S")
                row = html.Tr(
                    [
                        html.Td(timestamp),
                        html.Td(f"Class {pred['prediction_value']}"),
                        html.Td(f"{pred['confidence']:.3f}"),
                        html.Td(f"{pred['processing_time']:.3f}s"),
                    ]
                )
                table_rows.append(row)

            table = html.Table(
                [
                    html.Thead(
                        [
                            html.Tr(
                                [
                                    html.Th("Time"),
                                    html.Th("Prediction"),
                                    html.Th("Confidence"),
                                    html.Th("Processing Time"),
                                ]
                            )
                        ]
                    ),
                    html.Tbody(table_rows),
                ],
                className="data-table",
            )

            return table

        @self.app.callback(Output("alerts-table", "children"), [Input("alert-data", "data")])
        def update_alerts_table(alert_data):
            """Update alerts table."""
            if not alert_data:
                return html.Div("No active alerts")

            # Get unacknowledged alerts
            active_alerts = [alert for alert in alert_data if not alert["acknowledged"]]

            if not active_alerts:
                return html.Div("No active alerts", className="no-alerts")

            table_rows = []
            for alert in active_alerts[-10:]:  # Last 10 alerts
                timestamp = datetime.fromisoformat(alert["timestamp"]).strftime("%m-%d %H:%M")
                severity_class = f"severity-{alert['severity']}"

                row = html.Tr(
                    [
                        html.Td(timestamp),
                        html.Td(alert["alert_type"]),
                        html.Td(alert["severity"], className=severity_class),
                        html.Td(alert["message"]),
                    ],
                    className=severity_class,
                )
                table_rows.append(row)

            table = html.Table(
                [
                    html.Thead(
                        [
                            html.Tr(
                                [
                                    html.Th("Time"),
                                    html.Th("Type"),
                                    html.Th("Severity"),
                                    html.Th("Message"),
                                ]
                            )
                        ]
                    ),
                    html.Tbody(table_rows),
                ],
                className="data-table alerts-table",
            )

            return table

    def _background_monitoring(self):
        """Background thread for continuous monitoring."""
        while True:
            try:
                # Monitor system health
                self._collect_system_health()

                # Monitor for drift (simulated)
                self._collect_drift_metrics()

                # Clean old data
                self._cleanup_old_data()

            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")

            time.sleep(self.refresh_interval)

    def _collect_system_health(self):
        """Collect system health metrics."""
        try:
            # Health check
            response = requests.get(f"{self.model_server_url}/health", timeout=10)
            health_data = response.json()

            # Model info
            model_response = requests.get(f"{self.model_server_url}/model_info", timeout=10)

            conn = sqlite3.connect(self.monitoring_db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO system_health
                (timestamp, server_status, response_time, memory_usage, cpu_usage, model_loaded)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    health_data.get("status", "unknown"),
                    response.elapsed.total_seconds(),
                    0.0,  # Would get actual memory usage in production
                    0.0,  # Would get actual CPU usage in production
                    health_data.get("model_loaded", False),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to collect system health: {e}")

            # Log unhealthy status
            conn = sqlite3.connect(self.monitoring_db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO system_health
                (timestamp, server_status, response_time, memory_usage, cpu_usage, model_loaded)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (datetime.now().isoformat(), "unhealthy", 999.0, 0.0, 0.0, False),
            )

            conn.commit()
            conn.close()

    def _collect_drift_metrics(self):
        """Collect simulated drift metrics."""
        import random

        conn = sqlite3.connect(self.monitoring_db_path)
        cursor = conn.cursor()

        # Simulate drift detection
        drift_score = random.uniform(0.0, 0.3)
        drift_detected = drift_score > 0.1
        data_quality_score = random.uniform(0.85, 1.0)

        cursor.execute(
            """
            INSERT INTO drift_metrics
            (timestamp, drift_detected, drift_score, features_with_drift, data_quality_score, drift_method)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                drift_detected,
                drift_score,
                random.randint(0, 5),
                data_quality_score,
                "kolmogorov_smirnov",
            ),
        )

        # Generate alert if drift detected
        if drift_detected:
            cursor.execute(
                """
                INSERT INTO alerts
                (timestamp, alert_type, severity, message, acknowledged)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    "data_drift",
                    "medium",
                    f"Data drift detected with score {drift_score:.3f}",
                    False,
                ),
            )

        conn.commit()
        conn.close()

    def _get_system_health_data(self) -> List[Dict]:
        """Get system health data from database."""
        conn = sqlite3.connect(self.monitoring_db_path)

        # Get last 24 hours of data
        query = """
            SELECT timestamp, server_status, response_time, memory_usage, cpu_usage, model_loaded
            FROM system_health
            WHERE timestamp > datetime('now', '-24 hours')
            ORDER BY timestamp
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df.to_dict("records")

    def _get_prediction_data(self) -> List[Dict]:
        """Get prediction data from database."""
        conn = sqlite3.connect(self.monitoring_db_path)

        # Get last 24 hours of data
        query = """
            SELECT timestamp, prediction_value, confidence, processing_time
            FROM predictions
            WHERE timestamp > datetime('now', '-24 hours')
            ORDER BY timestamp
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df.to_dict("records")

    def _get_drift_data(self) -> List[Dict]:
        """Get drift data from database."""
        conn = sqlite3.connect(self.monitoring_db_path)

        # Get last 7 days of data
        query = """
            SELECT timestamp, drift_detected, drift_score, features_with_drift, data_quality_score
            FROM drift_metrics
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df.to_dict("records")

    def _get_alert_data(self) -> List[Dict]:
        """Get alert data from database."""
        conn = sqlite3.connect(self.monitoring_db_path)

        # Get last 7 days of alerts
        query = """
            SELECT timestamp, alert_type, severity, message, acknowledged
            FROM alerts
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df.to_dict("records")

    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        conn = sqlite3.connect(self.monitoring_db_path)
        cursor = conn.cursor()

        # Keep only last 30 days of data
        cursor.execute("DELETE FROM system_health WHERE timestamp < datetime('now', '-30 days')")
        cursor.execute("DELETE FROM predictions WHERE timestamp < datetime('now', '-30 days')")
        cursor.execute("DELETE FROM drift_metrics WHERE timestamp < datetime('now', '-30 days')")
        cursor.execute("DELETE FROM alerts WHERE timestamp < datetime('now', '-30 days')")

        conn.commit()
        conn.close()

    def run(self, host="0.0.0.0", port=8050, debug=False):
        """Run the dashboard."""
        logger.info(f"Starting monitoring dashboard on http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


def main():
    """Main function to run the dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="MLOps Monitoring Dashboard")
    parser.add_argument(
        "--model_server_url", default="http://localhost:5001", help="URL of the model server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--refresh_interval", type=int, default=30, help="Refresh interval in seconds"
    )

    args = parser.parse_args()

    # Add custom CSS
    custom_css = """
    .dashboard-container {
        font-family: 'Arial', sans-serif;
        margin: 0;
        padding: 20px;
        background-color: #f5f5f5;
    }

    .header {
        background-color: #1f2937;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    .dashboard-title {
        margin: 0;
        font-size: 2.5em;
    }

    .dashboard-subtitle {
        margin: 10px 0 0 0;
        opacity: 0.8;
    }

    .last-update {
        font-size: 0.9em;
        opacity: 0.7;
        margin-top: 10px;
    }

    .metrics-row {
        display: flex;
        gap: 20px;
        margin-bottom: 20px;
    }

    .metric-box {
        flex: 1;
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .metric-box h3 {
        margin: 0 0 10px 0;
        color: #374151;
    }

    .status-text {
        font-size: 1.2em;
        font-weight: bold;
    }

    .detail-text {
        font-size: 0.9em;
        color: #6b7280;
    }

    .metric-number {
        font-size: 2em;
        font-weight: bold;
        color: #1f2937;
    }

    .metric-unit {
        color: #6b7280;
    }

    .charts-row {
        display: flex;
        gap: 20px;
        margin-bottom: 20px;
    }

    .chart-container {
        flex: 1;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .tables-row {
        display: flex;
        gap: 20px;
    }

    .table-container {
        flex: 1;
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .table-container h3 {
        margin: 0 0 15px 0;
        color: #374151;
    }

    .data-table {
        width: 100%;
        border-collapse: collapse;
    }

    .data-table th, .data-table td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid #e5e7eb;
    }

    .data-table th {
        background-color: #f9fafb;
        font-weight: bold;
        color: #374151;
    }

    .severity-high {
        background-color: #fef2f2;
        color: #dc2626;
    }

    .severity-medium {
        background-color: #fffbeb;
        color: #d97706;
    }

    .severity-low {
        background-color: #f0fdf4;
        color: #16a34a;
    }

    .no-alerts {
        color: #16a34a;
        font-style: italic;
    }
    """

    # Create dashboard
    dashboard = MLOpsDashboard(
        model_server_url=args.model_server_url, refresh_interval=args.refresh_interval
    )

    # Add custom CSS to the app
    dashboard.app.index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                {custom_css}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    """

    # Run dashboard
    dashboard.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

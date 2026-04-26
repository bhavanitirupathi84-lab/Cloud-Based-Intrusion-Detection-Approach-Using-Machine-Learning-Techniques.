```python
# app.py
# Dashboard for Intrusion Detection System

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
import pickle
import pandas as pd

# Load results
print("Loading results...")
with open('data/results/dual_model_results.pkl', 'rb') as f:
    results = pickle.load(f)

y_pred = results['y_pred']
y_test = results['y_test']
accuracy = results['accuracy']
cm = results['confusion_matrix']

category_names = [
    'Normal', 'Brute Force', 'Data Exfiltration',
    'Geo Anomaly', 'Privilege Escalation', 'Insider Threat'
]

# Calculate statistics
total_events = len(y_test)
total_attacks = np.sum(y_test != 0)
detected_attacks = np.sum((y_pred == y_test) & (y_test != 0))
detection_rate = detected_attacks / total_attacks * 100 if total_attacks > 0 else 0

# Attack distribution
attack_counts = []
for i in range(6):
    count = np.sum(y_pred == i)
    attack_counts.append(count)

print(f"✓ Loaded {total_events:,} events")
print(f"✓ Detection rate: {detection_rate:.2f}%")

# Create Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🛡️ Cloud Intrusion Detection System", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.H3("Dual-Model ML Pipeline (Isolation Forest + Random Forest)", 
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': 0})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Metrics Row
    html.Div([
        # Metric 1
        html.Div([
            html.H4("Total Events", style={'color': '#7f8c8d', 'marginBottom': 5}),
            html.H2(f"{total_events:,}", style={'color': '#2c3e50', 'marginTop': 0})
        ], style={'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                  'backgroundColor': '#fff', 'padding': '20px', 'margin': '1%',
                  'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        # Metric 2
        html.Div([
            html.H4("Attacks Detected", style={'color': '#7f8c8d', 'marginBottom': 5}),
            html.H2(f"{detected_attacks:,}", style={'color': '#e74c3c', 'marginTop': 0})
        ], style={'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                  'backgroundColor': '#fff', 'padding': '20px', 'margin': '1%',
                  'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        # Metric 3
        html.Div([
            html.H4("Detection Rate", style={'color': '#7f8c8d', 'marginBottom': 5}),
            html.H2(f"{detection_rate:.1f}%", style={'color': '#27ae60', 'marginTop': 0})
        ], style={'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                  'backgroundColor': '#fff', 'padding': '20px', 'margin': '1%',
                  'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        # Metric 4
        html.Div([
            html.H4("Overall Accuracy", style={'color': '#7f8c8d', 'marginBottom': 5}),
            html.H2(f"{accuracy*100:.1f}%", style={'color': '#3498db', 'marginTop': 0})
        ], style={'width': '23%', 'display': 'inline-block', 'textAlign': 'center',
                  'backgroundColor': '#fff', 'padding': '20px', 'margin': '1%',
                  'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    ], style={'marginBottom': '20px'}),
    
    # Charts Row 1
    html.Div([
        # Attack Distribution Pie Chart
        html.Div([
            dcc.Graph(
                id='attack-distribution',
                figure={
                    'data': [
                        go.Pie(
                            labels=category_names,
                            values=attack_counts,
                            hole=0.3,
                            marker=dict(colors=['#27ae60', '#e74c3c', '#e67e22', 
                                               '#f39c12', '#9b59b6', '#c0392b'])
                        )
                    ],
                    'layout': go.Layout(
                        title='Attack Type Distribution',
                        showlegend=True
                    )
                }
            )
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
        
        # Confusion Matrix Heatmap
        html.Div([
            dcc.Graph(
                id='confusion-matrix',
                figure={
                    'data': [
                        go.Heatmap(
                            z=cm,
                            x=category_names,
                            y=category_names,
                            colorscale='RdYlGn',
                            text=cm,
                            texttemplate='%{text}',
                            textfont={"size": 10}
                        )
                    ],
                    'layout': go.Layout(
                        title='Confusion Matrix',
                        xaxis={'title': 'Predicted'},
                        yaxis={'title': 'Actual'}
                    )
                }
            )
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
    ], style={'marginBottom': '20px'}),
    
    # Charts Row 2
    html.Div([
        # Per-Class Performance Bar Chart
        html.Div([
            dcc.Graph(
                id='class-performance',
                figure={
                    'data': [
                        go.Bar(
                            x=category_names[1:],  # Skip normal
                            y=[
                                np.sum((y_pred == i) & (y_test == i)) / np.sum(y_test == i) * 100
                                if np.sum(y_test == i) > 0 else 0
                                for i in range(1, 6)
                            ],
                            marker=dict(color=['#e74c3c', '#e67e22', '#f39c12', '#9b59b6', '#c0392b'])
                        )
                    ],
                    'layout': go.Layout(
                        title='Detection Rate by Attack Type',
                        xaxis={'title': 'Attack Type'},
                        yaxis={'title': 'Detection Rate (%)', 'range': [0, 100]}
                    )
                }
            )
        ], style={'width': '98%', 'display': 'inline-block', 'margin': '1%'}),
    ]),
    
    # Footer
    html.Div([
        html.P("Intrusion Detection System using Isolation Forest + Random Forest",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': 20})
    ])
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f5f6fa', 'padding': '20px'})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 STARTING DASHBOARD")
    print("="*70)
    print("\nDashboard is running at: http://127.0.0.1:8050/")
    print("\nPress Ctrl+C to stop the dashboard")
    print("="*70 + "\n")
    
    app.run_server(debug=True, use_reloader=False)
```

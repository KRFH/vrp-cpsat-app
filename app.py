import dash
from dash import dcc, html
import plotly.graph_objects as go
from vrp_solver import solve_vrp

# Coordinates for visualization purposes
COORDS = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 1),
    3: (1, 0),
}

status, obj, routes = solve_vrp()

# Build figure with one trace per vehicle
fig = go.Figure()
colors = ["red", "blue", "green", "orange", "purple"]
for idx, (veh, route) in enumerate(routes.items()):
    xs = [COORDS[n][0] for n in route]
    ys = [COORDS[n][1] for n in route]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                             name=f"Vehicle {veh}",
                             line=dict(color=colors[idx % len(colors)])))

fig.update_layout(title="VRP Solution", xaxis_title="X", yaxis_title="Y")

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig),
    html.Div(f"Total distance: {obj}")
])

if __name__ == "__main__":
    app.run_server(debug=True)

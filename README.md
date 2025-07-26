# vrp-cpsat-app

This project provides a minimal CP-SAT based vehicle routing problem (VRP) solver
and a simple Dash UI to visualise the resulting routes.

## Running the solver

```
python vrp_solver.py
```

## Running the Dash app

```
python app.py
```

The Dash application displays the routes for each vehicle on a small grid using
Plotly. The optimisation logic is isolated in `vrp_solver.py` so the UI and
solver remain independent.

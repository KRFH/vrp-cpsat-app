"""
Minimal VRP with CP-SAT (OR-Tools) — 2 vehicles, 5 customers, 1 depot, with Soft Time Windows (Fixed)
---------------------------------------------------------------------
Requirements implemented:
 - Vehicle departs/returns from depot
 - Flow conservation per vehicle
 - Each customer visited exactly once
 - MTZ subtour elimination
 - Soft time windows (slack vars) and service times

Note: Time propagation to the depot return arcs has been disabled to avoid infeasibility from cycle constraints.
"""

from ortools.sat.python import cp_model

# ---------------------- Problem data ----------------------
# Nodes: 0 is depot, 1..5 are customers
NODES = [0, 1, 2, 3, 4, 5]
DEPOT = 0
CUSTOMERS = [1, 2, 3, 4, 5]

# Vehicles
VEHICLES = [0, 1]

# Distance matrix (travel times)
D = [
    [0, 10, 15, 20, 18, 25],
    [10, 0, 35, 25, 20, 28],
    [15, 35, 0, 30, 26, 32],
    [20, 25, 30, 0, 14, 22],
    [18, 20, 26, 14, 0, 19],
    [25, 28, 32, 22, 19, 0],
]
# Service times
service_time = {i: 5 for i in NODES}
service_time[0] = 0

# Time windows (earliest, latest)
time_windows = {
    0: (0, 200),
    1: (10, 50),
    2: (20, 60),
    3: (15, 55),
    4: (0, 40),
    5: (30, 80),
}
max_window = max(b for (_, b) in time_windows.values())
BIG_M = max_window + max(service_time.values())

# MTZ rank max
max_rank = len(CUSTOMERS)
# Penalty weight for time violations
time_penalty = 100

# ---------------------- Model ----------------------
model = cp_model.CpModel()

# Route decision variables
x = {}
for i in NODES:
    for j in NODES:
        if i == j:
            continue
        for k in VEHICLES:
            x[i, j, k] = model.NewBoolVar(f"x_{i}_{j}_v{k}")
# Aggregate arc usage for MTZ
tot = {}
for i in NODES:
    for j in NODES:
        if i == j:
            continue
        tot[i, j] = model.NewBoolVar(f"x_tot_{i}_{j}")
        model.AddMaxEquality(tot[i, j], [x[i, j, k] for k in VEHICLES])
# MTZ ordering
u = {i: model.NewIntVar(1, max_rank, f"u_{i}") for i in CUSTOMERS}
# Time and slack vars
t = {i: model.NewIntVar(0, BIG_M * 2, f"t_{i}") for i in NODES}
e = {i: model.NewIntVar(0, BIG_M, f"e_{i}") for i in NODES}
l = {i: model.NewIntVar(0, BIG_M, f"l_{i}") for i in NODES}

# ---------------------- Constraints ----------------------
# 1) Depot depart/return per vehicle
for k in VEHICLES:
    model.Add(sum(x[DEPOT, j, k] for j in NODES if j != DEPOT) == 1)
    model.Add(sum(x[i, DEPOT, k] for i in NODES if i != DEPOT) == 1)
# 2) Flow conservation at customers per vehicle
for k in VEHICLES:
    for i in CUSTOMERS:
        model.Add(sum(x[j, i, k] for j in NODES if j != i) == sum(x[i, j, k] for j in NODES if j != i))
# 3) Each customer visited exactly once
for i in CUSTOMERS:
    model.Add(sum(x[j, i, k] for j in NODES if j != i for k in VEHICLES) == 1)
# 4) MTZ subtour elimination
for i in CUSTOMERS:
    for j in CUSTOMERS:
        if i != j:
            model.Add(u[i] - u[j] + max_rank * tot[i, j] <= max_rank - 1)


# 5) Soft time windows at nodes
def add_time_window(i):
    a, b = time_windows[i]
    model.Add(t[i] + e[i] >= a)
    model.Add(t[i] - l[i] <= b)


for i in NODES:
    add_time_window(i)
# 6) Time propagation on used arcs (exclude return to depot)
for k in VEHICLES:
    for i in NODES:
        for j in NODES:
            if i == j or j == DEPOT:
                continue
            model.Add(t[j] >= t[i] + service_time[i] + D[i][j] - BIG_M * (1 - x[i, j, k]))

# ---------------------- Objective ----------------------
dist_cost = sum(D[i][j] * x[i, j, k] for i in NODES for j in NODES if i != j for k in VEHICLES)
time_cost = sum(e[i] + l[i] for i in NODES) * time_penalty
model.Minimize(dist_cost + time_cost)

# ---------------------- Solve ----------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 20
solver.parameters.num_search_workers = 8
status = solver.Solve(model)

# ---------------------- Output ----------------------
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f"Status: {solver.StatusName(status)}")
    print(f"Obj (dist + time penalties): {solver.ObjectiveValue()}\n")
    for k in VEHICLES:
        route = [DEPOT]
        cur = DEPOT
        while True:
            nxts = [j for j in NODES if j != cur and solver.Value(x[cur, j, k])]
            if not nxts:
                break
            cur = nxts[0]
            route.append(cur)
            if cur == DEPOT:
                break
        print(f"Vehicle {k} route: {route}")
    print("\nTime arrival & slacks:")
    for i in NODES:
        print(f" Node {i}: t={solver.Value(t[i])}, e={solver.Value(e[i])}, l={solver.Value(l[i])}")
else:
    print("No feasible solution found.")

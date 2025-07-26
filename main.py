"""
Minimal VRP with CP-SAT (OR-Tools) — 2 vehicles, 5 customers, 1 depot, with Soft Time Windows, Workers & Skills
-------------------------------------------------------------------------------------------------------
Requirements implemented:
 - Vehicle departs/returns from depot
 - Flow conservation per vehicle
 - Each customer visited exactly once
 - MTZ subtour elimination
 - Soft time windows and service times
 - Multiple workers assignment and boarding
 - Skill requirements at each customer
"""

from ortools.sat.python import cp_model

# ---------------------- Problem data ----------------------
# Nodes: 0 is depot, 1..5 customers
NODES = [0, 1, 2, 3, 4, 5]
DEPOT = 0
CUSTOMERS = [1, 2, 3, 4, 5]

# Vehicles
VEHICLES = [0, 1]
# Workers
WORKERS = [0, 1]  # two workers

# Skills and worker-skills mapping
SKILLS = ["electric", "plumbing"]
worker_skills = {
    0: ["electric"],
    1: ["plumbing", "electric"],
}
# Skill requirements per customer: req[i][s] = required number of workers with skill s at customer i
req = {
    1: {"electric": 1, "plumbing": 0},
    2: {"electric": 0, "plumbing": 1},
    3: {"electric": 1, "plumbing": 1},
    4: {"electric": 1, "plumbing": 0},
    5: {"electric": 0, "plumbing": 1},
}

# Travel times (distance)
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
time_windows = {0: (0, 200), 1: (10, 50), 2: (20, 60), 3: (15, 55), 4: (0, 40), 5: (30, 80)}
max_window = max(b for (_, b) in time_windows.values())
BIG_M = max_window + max(service_time.values())
# MTZ rank max
max_rank = len(CUSTOMERS)
# Penalty weight for time windows
time_penalty = 100

# ---------------------- Model ----------------------
model = cp_model.CpModel()

# 1) Route decision: x[i,j,k] = vehicle k travels i->j
x = {}
for i in NODES:
    for j in NODES:
        if i == j:
            continue
        for k in VEHICLES:
            x[i, j, k] = model.NewBoolVar(f"x_{i}_{j}_v{k}")
# 2) Aggregate arc usage for MTZ
tot = {}
for i in NODES:
    for j in NODES:
        if i == j:
            continue
        tot[i, j] = model.NewBoolVar(f"x_tot_{i}_{j}")
        model.AddMaxEquality(tot[i, j], [x[i, j, k] for k in VEHICLES])
# 3) MTZ order vars for customers
u = {i: model.NewIntVar(1, max_rank, f"u_{i}") for i in CUSTOMERS}
# 4) Time and slack vars
t = {i: model.NewIntVar(0, BIG_M * 2, f"t_{i}") for i in NODES}
e = {i: model.NewIntVar(0, BIG_M, f"e_{i}") for i in NODES}
l = {i: model.NewIntVar(0, BIG_M, f"l_{i}") for i in NODES}
# 5) Worker assignment and boarding
y = {}
p = {}
z = {}
for i in CUSTOMERS:
    for w in WORKERS:
        y[i, w] = model.NewBoolVar(f"y_{i}_{w}")
        for k in VEHICLES:
            p[i, w, k] = model.NewBoolVar(f"p_{i}_{w}_v{k}")
# Node-vehicle visit flag
for i in NODES:
    for k in VEHICLES:
        z[i, k] = model.NewBoolVar(f"z_{i}_v{k}")
        model.AddMaxEquality(z[i, k], [x[j, i, k] for j in NODES if j != i])

# ---------------------- Constraints ----------------------
# A) Depot depart/return per vehicle
for k in VEHICLES:
    model.Add(sum(x[DEPOT, j, k] for j in NODES if j != DEPOT) == 1)
    model.Add(sum(x[i, DEPOT, k] for i in NODES if i != DEPOT) == 1)
# B) Flow conservation at customers per vehicle
for k in VEHICLES:
    for i in CUSTOMERS:
        model.Add(sum(x[j, i, k] for j in NODES if j != i) == sum(x[i, j, k] for j in NODES if j != i))
# C) Each customer visited once by some vehicle
for i in CUSTOMERS:
    model.Add(sum(x[j, i, k] for j in NODES if j != i for k in VEHICLES) == 1)
# D) MTZ subtour elimination
for i in CUSTOMERS:
    for j in CUSTOMERS:
        if i != j:
            model.Add(u[i] - u[j] + max_rank * tot[i, j] <= max_rank - 1)
# E) Soft time windows
for i in NODES:
    a, b = time_windows[i]
    model.Add(t[i] + e[i] >= a)
    model.Add(t[i] - l[i] <= b)
# F) Time propagation (no return to depot)
for k in VEHICLES:
    for i in NODES:
        for j in NODES:
            if i == j or j == DEPOT:
                continue
            model.Add(t[j] >= t[i] + service_time[i] + D[i][j] - BIG_M * (1 - x[i, j, k]))
# G) Worker assignment: at least one per customer
for i in CUSTOMERS:
    model.Add(sum(y[i, w] for w in WORKERS) >= 1)
# H) Boarding consistency: if y[i,w]=1 then board exactly one vehicle and only if it visits
for i in CUSTOMERS:
    for w in WORKERS:
        model.Add(sum(p[i, w, k] for k in VEHICLES) == y[i, w])
        for k in VEHICLES:
            model.Add(p[i, w, k] <= z[i, k])
# I) Skill requirements: ensure enough skilled workers at each customer
for i in CUSTOMERS:
    for s in SKILLS:
        model.Add(sum(y[i, w] for w in WORKERS if s in worker_skills[w]) >= req[i][s])

# ---------------------- Objective ----------------------
# Minimize distance + time slack penalties
dist_cost = sum(D[i][j] * x[i, j, k] for i in NODES for j in NODES if i != j for k in VEHICLES)
time_cost = sum(e[i] + l[i] for i in NODES) * time_penalty
model.Minimize(dist_cost + time_cost)

# ---------------------- Solve ----------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30
solver.parameters.num_search_workers = 8
status = solver.Solve(model)

# ---------------------- Output ----------------------
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f"Status: {solver.StatusName(status)}")
    print(f"Obj: {solver.ObjectiveValue()}\n")
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
        print(f"Vehicle {k}: {[{r:solver.Value(t[r])} for r in route]}")
    print("\nAssignments and boarding:")
    for i in CUSTOMERS:
        for w in WORKERS:
            if solver.Value(y[i, w]):
                boarded = [(k) for k in VEHICLES if solver.Value(p[i, w, k])]
                print(f"Customer {i}: worker {w}, boards vehicles {boarded}")
else:
    print("No feasible solution found.")

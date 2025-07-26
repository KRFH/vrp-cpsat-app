"""
Minimal VRP with CP-SAT (OR-Tools) — 2 vehicles, 5 customers, 1 depot, Enhanced Workers with Transfers & Labor Limits
-----------------------------------------------------------------------------------------------------------
Requirements implemented:
 - Vehicle departs/returns from depot
 - Flow conservation per vehicle
 - Each customer visited exactly once
 - MTZ subtour elimination
 - Soft time windows and service times
 - Multiple workers assignment and boarding with transfers
 - Skill requirements at each customer
 - Worker labor time limits

Note: Removed single-vehicle assignment (yv). Workers can board multiple vehicles (transfers). Labor time counts service only; travel-time inclusion can be added.
"""

from ortools.sat.python import cp_model

# ---------------------- Problem data ----------------------
NODES = [0, 1, 2, 3, 4, 5]  # 0=depot, 1..5 customers
DEPOT = 0
CUSTOMERS = [1, 2, 3, 4, 5]
VEHICLES = [0, 1]
WORKERS = [0, 1]

# Skills
SKILLS = ["electric", "plumbing"]
worker_skills = {0: ["electric"], 1: ["plumbing", "electric"]}
req = {
    1: {"electric": 1, "plumbing": 0},
    2: {"electric": 0, "plumbing": 1},
    3: {"electric": 1, "plumbing": 1},
    4: {"electric": 1, "plumbing": 0},
    5: {"electric": 0, "plumbing": 1},
}

# Travel & service
distance = [
    [0, 10, 15, 20, 18, 25],
    [10, 0, 35, 25, 20, 28],
    [15, 35, 0, 30, 26, 32],
    [20, 25, 30, 0, 14, 22],
    [18, 20, 26, 14, 0, 19],
    [25, 28, 32, 22, 19, 0],
]
service_time = {i: 5 for i in NODES}
service_time[0] = 0
# Time windows
time_windows = {0: (0, 200), 1: (10, 50), 2: (20, 60), 3: (15, 55), 4: (0, 40), 5: (30, 80)}
BIG_M = max(b for (_, b) in time_windows.values()) + max(service_time.values())
# MTZ
max_rank = len(CUSTOMERS)
time_penalty = 100
# Labor limit (minutes)
LABOR_LIMIT = 20

# ---------------------- Model ----------------------
model = cp_model.CpModel()

# 1) Vehicle arcs x[i,j,k]
x = {}
for i in NODES:
    for j in NODES:
        if i == j:
            continue
        for k in VEHICLES:
            x[i, j, k] = model.NewBoolVar(f"x_{i}_{j}_v{k}")
# 2) Aggregated arc tot for MTZ
tot = {}
for i in NODES:
    for j in NODES:
        if i == j:
            continue
        tot[i, j] = model.NewBoolVar(f"x_tot_{i}_{j}")
        model.AddMaxEquality(tot[i, j], [x[i, j, k] for k in VEHICLES])
# 3) MTZ order vars u[i]
u = {i: model.NewIntVar(1, max_rank, f"u_{i}") for i in CUSTOMERS}
# 4) Time & slack vars
t = {i: model.NewIntVar(0, BIG_M * 2, f"t_{i}") for i in NODES}
e = {i: model.NewIntVar(0, BIG_M, f"e_{i}") for i in NODES}
l = {i: model.NewIntVar(0, BIG_M, f"l_{i}") for i in NODES}
# 5) Workers & boarding y[i,w], p[i,w,k]
y = {}
p = {}
for i in CUSTOMERS:
    for w in WORKERS:
        y[i, w] = model.NewBoolVar(f"y_{i}_{w}")
        for k in VEHICLES:
            p[i, w, k] = model.NewBoolVar(f"p_{i}_{w}_v{k}")
# Node-vehicle visit z[i,k]
z = {}
for i in NODES:
    for k in VEHICLES:
        z[i, k] = model.NewBoolVar(f"z_{i}_v{k}")
        model.AddMaxEquality(z[i, k], [x[j, i, k] for j in NODES if j != i])
# 6) Labor time L[w]
L = {w: model.NewIntVar(0, LABOR_LIMIT, f"L_{w}") for w in WORKERS}

# ---------------------- Constraints ----------------------
# A) Depot flow per vehicle
for k in VEHICLES:
    model.Add(sum(x[DEPOT, j, k] for j in NODES if j != DEPOT) == 1)
    model.Add(sum(x[i, DEPOT, k] for i in NODES if i != DEPOT) == 1)
# B) Flow conservation at customers
for k in VEHICLES:
    for i in CUSTOMERS:
        model.Add(sum(x[j, i, k] for j in NODES if j != i) == sum(x[i, j, k] for j in NODES if j != i))
# C) Each customer once
for i in CUSTOMERS:
    model.Add(sum(x[j, i, k] for j in NODES if j != i for k in VEHICLES) == 1)
# D) MTZ
for i in CUSTOMERS:
    for j in CUSTOMERS:
        if i != j:
            model.Add(u[i] - u[j] + max_rank * tot[i, j] <= max_rank - 1)


# E) Soft time windows
def add_time_window(i):
    a, b = time_windows[i]
    model.Add(t[i] + e[i] >= a)
    model.Add(t[i] - l[i] <= b)


for i in NODES:
    add_time_window(i)
# F) Time propagation (no return to depot)
for k in VEHICLES:
    for i in NODES:
        for j in NODES:
            if i == j or j == DEPOT:
                continue
            model.Add(t[j] >= t[i] + service_time[i] + distance[i][j] - BIG_M * (1 - x[i, j, k]))
# G) Skills & worker assignment
for i in CUSTOMERS:
    # at least one worker
    model.Add(sum(y[i, w] for w in WORKERS) >= 1)
    # skill coverage
    for s in SKILLS:
        model.Add(sum(y[i, w] for w in WORKERS if s in worker_skills[w]) >= req[i][s])
    # boarding constraints allow transfers
    for w in WORKERS:
        model.Add(sum(p[i, w, k] for k in VEHICLES) == y[i, w])
        for k in VEHICLES:
            model.Add(p[i, w, k] <= z[i, k])
# H) Labor time from service only
for w in WORKERS:
    model.Add(L[w] == sum(service_time[i] * y[i, w] for i in CUSTOMERS))
    model.Add(L[w] <= LABOR_LIMIT)

# ---------------------- Objective ----------------------
cost_dist = sum(distance[i][j] * x[i, j, k] for i in NODES for j in NODES if i != j for k in VEHICLES)
cost_time = sum(e[i] + l[i] for i in NODES) * time_penalty
model.Minimize(cost_dist + cost_time)

# ---------------------- Solve ----------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30
solver.parameters.num_search_workers = 8
res = solver.Solve(model)

# ---------------------- Output ----------------------
if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f"Status: {solver.StatusName(res)}")
    print(f"Obj: {solver.ObjectiveValue()}")
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
        print(f"Vehicle {k}: {route}")
    print("\nWorker labor times:")
    for w in WORKERS:
        print(f" Worker {w}: L[{w}] = {solver.Value(L[w])} <= {LABOR_LIMIT}")
    print("\nCustomer worker boarding:")
    for i in CUSTOMERS:
        for w in WORKERS:
            if solver.Value(y[i, w]):
                boards = [k for k in VEHICLES if solver.Value(p[i, w, k])]
                print(f" Customer {i}: worker {w} boards {boards}")
else:
    print("No feasible solution found.")

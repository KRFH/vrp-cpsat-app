"""
Minimal VRP with CP-SAT (OR-Tools) — 2 vehicles, 3 customers, 1 depot
---------------------------------------------------------------------
Requirements implemented:
 1. Each vehicle leaves the depot (0) exactly once and comes back exactly once.
 2. Flow conservation at every non-depot node for every vehicle (入=出)。
 3. Every customer is visited exactly once by some vehicle.
 4. Subtour elimination (MTZ) across the aggregated arcs so that all visited nodes connect to the depot.

NOT implemented (yet): workers, time windows, capacities, skills, etc.
We'll add them incrementally after you confirm this base is fine.

How to run:
    pip install ortools
    python minimal_cpsat_vrp_two_vehicles_three_nodes.py

"""

from ortools.sat.python import cp_model

# ---------------------- Problem data ----------------------
# Nodes: 0 is depot, 1..3 are customers
NODES = [0, 1, 2, 3]
DEPOT = 0
CUSTOMERS = [1, 2, 3]

# Vehicles
VEHICLES = [0, 1]  # two vehicles

# Symmetric distance matrix (km). Feel free to change.
#        0   1   2   3
D = [
    [0, 10, 15, 20],  # 0
    [10, 0, 35, 25],  # 1
    [15, 35, 0, 30],  # 2
    [20, 25, 30, 0],  # 3
]

# A large enough M for MTZ (n = number of nodes)
N = len(NODES)
BIG_M = N  # could also be N-1 or larger

# ---------------------- Model ----------------------
model = cp_model.CpModel()

# Binary arc decision: x[i][j][k] = 1 if vehicle k travels i -> j
x = {}
for i in NODES:
    for j in NODES:
        if i == j:
            continue
        for k in VEHICLES:
            x[i, j, k] = model.NewBoolVar(f"x_{i}_{j}_v{k}")

# Helper: total arc usage regardless of vehicle (for MTZ)
x_tot = {}
for i in NODES:
    for j in NODES:
        if i == j:
            continue
        x_tot[i, j] = model.NewBoolVar(f"x_tot_{i}_{j}")
        # x_tot = OR over vehicles
        model.AddMaxEquality(x_tot[i, j], [x[i, j, k] for k in VEHICLES])

# MTZ order variables for customers only (not for depot)
# u_i in [1, |CUSTOMERS|]
max_rank = len(CUSTOMERS)
u = {}
for i in CUSTOMERS:
    u[i] = model.NewIntVar(1, max_rank, f"u_{i}")

# ---------------------- Constraints ----------------------
# 1) Each vehicle: exactly one edge leaves depot and one enters depot
for k in VEHICLES:
    model.Add(sum(x[DEPOT, j, k] for j in NODES if j != DEPOT) == 1)
    model.Add(sum(x[i, DEPOT, k] for i in NODES if i != DEPOT) == 1)

# 2) Flow conservation at customers for each vehicle
for k in VEHICLES:
    for i in CUSTOMERS:
        model.Add(sum(x[j, i, k] for j in NODES if j != i) == sum(x[i, j, k] for j in NODES if j != i))

# 3) Each customer visited exactly once by some vehicle
for i in CUSTOMERS:
    model.Add(sum(x[j, i, k] for j in NODES if j != i for k in VEHICLES) == 1)

# 4) MTZ subtour elimination on aggregated arcs (classic form for TSP, adapted):
#    u_i - u_j + (max_rank) * x_tot[i,j] <= max_rank - 1,  for all i != j, i,j in CUSTOMERS
for i in CUSTOMERS:
    for j in CUSTOMERS:
        if i == j:
            continue
        model.Add(u[i] - u[j] + max_rank * x_tot[i, j] <= max_rank - 1)

# ---------------------- Objective ----------------------
# Minimize total distance traveled
model.Minimize(sum(D[i][j] * x[i, j, k] for i in NODES for j in NODES if i != j for k in VEHICLES))

# ---------------------- Solve ----------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 10
solver.parameters.num_search_workers = 8
status = solver.Solve(model)

# ---------------------- Output ----------------------
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f"Status: {solver.StatusName(status)}")
    print(f"Objective (total distance): {solver.ObjectiveValue()}\n")

    # Reconstruct routes per vehicle
    for k in VEHICLES:
        print(f"Vehicle {k} route:")
        # start from depot
        current = DEPOT
        route = [current]
        while True:
            next_nodes = [j for j in NODES if j != current and solver.Value(x[current, j, k]) == 1]
            if not next_nodes:
                break
            nxt = next_nodes[0]
            route.append(nxt)
            current = nxt
            if current == DEPOT:
                break
        print("  ", " -> ".join(map(str, route)))
    print("\nOrder variables (u_i):")
    for i in CUSTOMERS:
        print(f"  u_{i} = {solver.Value(u[i])}")
else:
    print("No feasible solution found.")

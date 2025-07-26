from ortools.sat.python import cp_model

# Problem data
NODES = [0, 1, 2, 3]
DEPOT = 0
CUSTOMERS = [1, 2, 3]
VEHICLES = [0, 1]
D = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]


def solve_vrp():
    """Solve VRP and return objective and routes per vehicle."""
    model = cp_model.CpModel()

    # Decision variables
    x = {}
    for i in NODES:
        for j in NODES:
            if i == j:
                continue
            for k in VEHICLES:
                x[i, j, k] = model.NewBoolVar(f"x_{i}_{j}_v{k}")

    x_tot = {}
    for i in NODES:
        for j in NODES:
            if i == j:
                continue
            x_tot[i, j] = model.NewBoolVar(f"x_tot_{i}_{j}")
            model.AddMaxEquality(x_tot[i, j], [x[i, j, k] for k in VEHICLES])

    max_rank = len(CUSTOMERS)
    u = {i: model.NewIntVar(1, max_rank, f"u_{i}") for i in CUSTOMERS}

    # Constraints
    for k in VEHICLES:
        model.Add(sum(x[DEPOT, j, k] for j in NODES if j != DEPOT) == 1)
        model.Add(sum(x[i, DEPOT, k] for i in NODES if i != DEPOT) == 1)

    for k in VEHICLES:
        for i in CUSTOMERS:
            model.Add(sum(x[j, i, k] for j in NODES if j != i) ==
                      sum(x[i, j, k] for j in NODES if j != i))

    for i in CUSTOMERS:
        model.Add(sum(x[j, i, k] for j in NODES if j != i for k in VEHICLES) == 1)

    for i in CUSTOMERS:
        for j in CUSTOMERS:
            if i == j:
                continue
            model.Add(u[i] - u[j] + max_rank * x_tot[i, j] <= max_rank - 1)

    # Objective
    model.Minimize(sum(D[i][j] * x[i, j, k]
                       for i in NODES for j in NODES if i != j
                       for k in VEHICLES))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    routes = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for k in VEHICLES:
            current = DEPOT
            route = [current]
            while True:
                next_nodes = [j for j in NODES if j != current and
                               solver.Value(x[current, j, k]) == 1]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                route.append(nxt)
                current = nxt
                if current == DEPOT:
                    break
            routes[k] = route
        obj = solver.ObjectiveValue()
    else:
        obj = None

    return status, obj, routes


if __name__ == "__main__":
    st, obj, routes = solve_vrp()
    print("Status:", st)
    print("Objective:", obj)
    print("Routes:", routes)

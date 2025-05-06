#task 1:



from ortools.sat.python import cp_model
import math

def warehouse_robot_navigation(grid, start, target):
    model = cp_model.CpModel()
    rows, cols = len(grid), len(grid[0])
    max_steps = rows + cols - 2
    diagonal_cost = int(math.sqrt(2) * 1000)
    
    robot_rows = []
    robot_cols = []
    active_steps = []
    
    for i in range(max_steps):
        robot_rows.append(model.NewIntVar(0, rows - 1, f'row_{i}'))
        robot_cols.append(model.NewIntVar(0, cols - 1, f'col_{i}'))
        active_steps.append(model.NewBoolVar(f'active_{i}'))
    
    model.Add(robot_rows[0] == start[0])
    model.Add(robot_cols[0] == start[1])
    
    for i in range(max_steps - 1):
        row_diff = model.NewIntVar(-1, 1, f'row_diff_{i}')
        col_diff = model.NewIntVar(-1, 1, f'col_diff_{i}')
        
        model.Add(row_diff == robot_rows[i + 1] - robot_rows[i])
        model.Add(col_diff == robot_cols[i + 1] - robot_cols[i])
        
        model.AddAbsEquality(1, row_diff)
        model.AddAbsEquality(1, col_diff)
        model.AddImplication(active_steps[i], active_steps[i + 1])
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                for i in range(max_steps):
                    model.AddForbiddenAssignments([robot_rows[i], robot_cols[i]], [(r, c)])
    
    total_cost = 0
    for i in range(max_steps):
        total_cost += active_steps[i] * diagonal_cost
    
    model.Minimize(total_cost)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        path = []
        for i in range(max_steps):
            if solver.Value(active_steps[i]):
                path.append((solver.Value(robot_rows[i]), solver.Value(robot_cols[i])))
        return path
    
    return None

grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]
start, target = (1, 1), (4, 4)

path = warehouse_robot_navigation(grid, start, target)
if path:
    print("Shortest Diagonal Path:", path)
else:
    print("No diagonal path found.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#task 2:



from ortools.sat.python import cp_model

def land_erosion_analysis(island_map):
    

    model = cp_model.CpModel()
    rows = len(island_map)
    cols = len(island_map[0])

    cells = [[model.NewBoolVar(f'cell_{r}_{c}') for c in range(cols)] for r in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if island_map[r][c] == 1:
                model.Add(cells[r][c] == 1)
            else:
                model.Add(cells[r][c] == 0)

    boundary_edges = 0
    for r in range(rows):
        for c in range(cols):
            if island_map[r][c] == 1:  # Land cell
                # Check adjacent cells (up, down, left, right)
                adjacent_water_count = 0

                # Check up
                if r > 0 and island_map[r - 1][c] == 0:
                    adjacent_water_count += 1
                # Check down
                if r < rows - 1 and island_map[r + 1][c] == 0:
                    adjacent_water_count += 1
                # Check left
                if c > 0 and island_map[r][c - 1] == 0:
                    adjacent_water_count += 1
                # Check right
                if c < cols - 1 and island_map[r][c + 1] == 0:
                    adjacent_water_count += 1

                boundary_edges += adjacent_water_count

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return boundary_edges
    else:
        return 0


island_map = [
    [0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [1, 1, 0, 0, 1]
]

perimeter = land_erosion_analysis(island_map)
print("Perimeter of the largest landmass:", perimeter) 




#task 3:


from ortools.sat.python import cp_model

def traveling_salesman_problem(distances):
    num_cities = len(distances)
    model = cp_model.CpModel()

    # Variables
    x = {}
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x[i, j] = model.NewBoolVar(f'x_{i}_{j}')

    for i in range(num_cities):
        model.Add(sum(x[i, j] for j in range(num_cities) if i != j) == 1)
        model.Add(sum(x[j, i] for j in range(num_cities) if i != j) == 1)

    u = [model.NewIntVar(0, num_cities - 1, f'u_{i}') for i in range(num_cities)]
    for i in range(1, num_cities):
        for j in range(1, num_cities):
            if i != j:
                model.Add(u[i] - u[j] + num_cities * x[i, j] <= num_cities - 1)

    total_distance = sum(distances[i][j] * x[i, j] for i in range(num_cities) for j in range(num_cities) if i != j)
    model.Minimize(total_distance)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        route = [0]  # Start from city 0
        current_city = 0
        visited = set([0])

        for _ in range(num_cities - 1):
            for j in range(num_cities):
                if j not in visited and solver.Value(x[current_city, j]) == 1:
                    route.append(j)
                    visited.add(j)
                    current_city = j
                    break

        route.append(0)  # Return to the starting city
        return route, solver.ObjectiveValue()
    else:
        return None, None


distances = [
    [0, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    [10, 0, 12, 18, 22, 28, 32, 38, 42, 48],
    [15, 12, 0, 8, 16, 24, 28, 34, 38, 44],
    [20, 18, 8, 0, 14, 20, 26, 32, 36, 42],
    [25, 22, 16, 14, 0, 12, 18, 24, 30, 36],
    [30, 28, 24, 20, 12, 0, 10, 16, 22, 28],
    [35, 32, 28, 26, 18, 10, 0, 8, 14, 20],
    [40, 38, 34, 32, 24, 16, 8, 0, 6, 12],
    [45, 42, 38, 36, 30, 22, 14, 6, 0, 10],
    [50, 48, 44, 42, 36, 28, 20, 12, 10, 0]
]

optimal_route, optimal_distance = traveling_salesman_problem(distances)

if optimal_route:
    print("Optimal Route:", optimal_route)
    print("Optimal Distance:", optimal_distance)
else:
    print("No solution found.")

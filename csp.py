"""
CSP Solutions Collection
Contains implementations for:
1. N-Queens Problem
2. Graph Coloring Problem
3. Job Scheduling Problem
"""

from ortools.sat.python import cp_model

# ====================== 1. N-Queens Problem ======================
def solve_n_queens(n=8):
    """Solves the N-queens problem using constraint programming."""
    model = cp_model.CpModel()
    
    # Each variable represents the row position of a queen in its column
    queens = [model.NewIntVar(0, n-1, f'q_{i}') for i in range(n)]
    
    # All queens must be in different rows
    model.AddAllDifferent(queens)
    
    # No two queens can be on the same diagonal
    diag1 = []
    diag2 = []
    for i in range(n):
        q1 = model.NewIntVar(-n, n, f'diag1_{i}')
        q2 = model.NewIntVar(0, 2*n, f'diag2_{i}')
        model.Add(q1 == queens[i] - i)
        model.Add(q2 == queens[i] + i)
        diag1.append(q1)
        diag2.append(q2)
    
    model.AddAllDifferent(diag1)
    model.AddAllDifferent(diag2)
    
    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"\nSolution for {n}-queens problem:")
        for i in range(n):
            row = ['_'] * n
            row[solver.Value(queens[i])] = 'Q'
            print(' '.join(row))
    else:
        print("No solution found.")

# ====================== 2. Graph Coloring Problem ======================
def solve_graph_coloring(adjacency_list, num_colors=3):
    """Solves graph coloring problem using constraint programming."""
    model = cp_model.CpModel()
    
    # Extract nodes from adjacency list
    nodes = list(adjacency_list.keys())
    
    # Create variables for each node's color
    colors = {}
    for node in nodes:
        colors[node] = model.NewIntVar(1, num_colors, f'color_{node}')
    
    # Add constraints: adjacent nodes must have different colors
    for node, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            model.Add(colors[node] != colors[neighbor])
    
    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("\nGraph coloring solution:")
        for node in nodes:
            print(f"Node {node}: Color {solver.Value(colors[node])}")
    else:
        print("No valid coloring found with the given number of colors.")

# ====================== 3. Job Scheduling Problem ======================
def solve_job_scheduling(jobs, num_machines=3, max_time=100):
    """Solves job scheduling problem to minimize makespan."""
    model = cp_model.CpModel()
    
    # Create variables for start and end times of each job
    starts = []
    ends = []
    intervals = []
    
    for i, (duration, machine) in enumerate(jobs):
        suffix = f"_{i}"
        start = model.NewIntVar(0, max_time, 'start' + suffix)
        end = model.NewIntVar(0, max_time, 'end' + suffix)
        interval = model.NewIntervalVar(start, duration, end, 'interval' + suffix)
        starts.append(start)
        ends.append(end)
        intervals.append(interval)
    
    # Create machine variables and constraints
    for machine_id in range(num_machines):
        machine_jobs = [
            intervals[i] for i in range(len(jobs)) 
            if jobs[i][1] == machine_id or jobs[i][1] is None
        ]
        if machine_jobs:
            model.AddNoOverlap(machine_jobs)
    
    # Objective: minimize the makespan (maximum end time)
    makespan = model.NewIntVar(0, max_time, 'makespan')
    model.AddMaxEquality(makespan, ends)
    model.Minimize(makespan)
    
    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL:
        print("\nOptimal schedule found:")
        print(f"Total duration: {solver.Value(makespan)}")
        for i, (duration, machine) in enumerate(jobs):
            print(f"Job {i}: starts at {solver.Value(starts[i])}, "
                  f"ends at {solver.Value(ends[i])}, "
                  f"machine {machine if machine is not None else 'any'}")
    else:
        print("No optimal solution found.")

# ====================== Example Usage ======================
if __name__ == "__main__":
    print("=== CSP Solutions Collection ===")
    
    # 1. N-Queens Example
    print("\nRunning N-Queens Problem (8 queens)...")
    solve_n_queens(8)
    
    # 2. Graph Coloring Example
    print("\nRunning Graph Coloring Problem (Australia map)...")
    australia = {
        'WA': ['NT', 'SA'],
        'NT': ['WA', 'SA', 'Q'],
        'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
        'Q': ['NT', 'SA', 'NSW'],
        'NSW': ['Q', 'SA', 'V'],
        'V': ['SA', 'NSW'],
        'T': []
    }
    solve_graph_coloring(australia, 3)
    
    # 3. Job Scheduling Example
    print("\nRunning Job Scheduling Problem...")
    jobs = [
        (3, 0),  # Job 0: duration 3, must run on machine 0
        (2, 1),  # Job 1: duration 2, must run on machine 1
        (4, None),  # Job 2: duration 4, can run on any machine
        (1, 0),  # Job 3: duration 1, must run on machine 0
        (5, None),  # Job 4: duration 5, can run on any machine
        (3, 2),  # Job 5: duration 3, must run on machine 2
    ]
    solve_job_scheduling(jobs, num_machines=3)
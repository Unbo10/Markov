import random
import pandas as pd
import numpy as np

# def print_possible_routes(states: list[int], current_state: int) -> None:
#     states: set = set(states)
#     current_states: list[int] = random.sample(sorted(states), current_state)
#     count: int = 0
#     for target1 in states:
#         for target2 in states:
#             if target1 != current_states[0] and target2 != current_states[1]:
#                 print(f"{current_states[0]} -> {target1} | {current_states[1]} -> {target2}")
#                 count += 1
#     print("Total routes:", count)

def sumit(it) -> float:
    count = 0
    for i in it:
        count += i
    return count

def possible_routes(states: list[int], current_state: int) -> tuple[list[list[int]], list[int]]:
    n_states: int = len(states)
    current_states: list[int] = random.sample(sorted(states), current_state)
    count: int = 0

    targets_list: list[list[int]] = [states for t in range(current_state, 2*current_state + 1)]
    counters: list[int] = [0 for _ in range(current_state)]
    # print(counters, len(counters))

    i: int = 0
    transitions: list[list[int]] = []
    while i < (n_states) ** current_state:
        for current in range(len(current_states)):
            to_print: str = ""
            valid: bool = True
            j: int = 0
            temp: list[int] = [] #*List to store the target values
            for c in counters:
                # print(c)
                if current_states[j] != targets_list[j][c]:
                    to_print += f"{current_states[j]} -> {targets_list[j][c]} | "
                    temp.append(targets_list[j][c])
                    j += 1
                else:
                    valid = False
            if valid:
                # print(to_print)
                transitions.append(temp)
                count += 1
            sum: int = 1
            # print("Sum:", sumit(counters), count)
            for c in range(current_state):
                counters[c] += sum
                if counters[c] >= n_states:
                    counters[c] = 0
                    sum = 1
                else:
                    sum = 0
            if sumit(counters) == 0:
                # print(count)
                return transitions, current_states
            i += 1
    # print(count)

def get_classes(routes: pd.DataFrame, current_states: list[int]) -> list[pd.DataFrame]:
    df = pd.DataFrame(routes, columns=[i for i in current_states])
    df['class'] = 0
    for row in range(len(routes)):
        new_states = set(current_states)
        for state in range(len(current_states)):
            target_state = routes.iloc[row, state]
            if target_state not in new_states:
                df.at[row, 'class'] += 1
                new_states.add(target_state)
    return df

    # for tlist in targets_list:
    #     for target in tlist:
    #         if target1 != current_states[0] and target2 != current_states[1]:
    #             print(f"{current_states[0]} -> {target1} | {current_states[1]} -> {target2}")
    #             count += 1

    # for target1 in states:
    #     for target2 in states:
    #         if target1 != current_states[0] and target2 != current_states[1]:
    #             print(f"{current_states[0]} -> {target1} | {current_states[1]} -> {target2}")
    #             count += 1
    # print("Total routes:", count)

def print_classes_summary(classes: pd.DataFrame) -> None:
    unique_classes = list(classes['class'].unique())
    unique_classes = sorted([int(x) for x in unique_classes])
    n_class_rows: list[int] = []
    for class_val in unique_classes:
        class_rows = classes[classes['class'] == class_val]
        n_class_rows.append(len(class_rows))
        print(f"Class {class_val}):")
        print(class_rows)
        print()
    print(unique_classes)
    for class_val in range(len(unique_classes)):
        print(f"Class {unique_classes[class_val]} has {n_class_rows[class_val]} elements")
    print(f"The total sum of elements in the {len(unique_classes)} classes is {sumit(n_class_rows)}")

def solve_system(A: np.array, b:np.array) -> np.array:
    solution: np.array = np.linalg.solve(a=A, b=b)
    return solution

def build_transition_matrix(n: int = 2) -> np.array:
    if n < 2:
        raise ValueError("n must be greater than or equal to 2")
    states: list[int] = [s for s in range(1, n + 1)]
    P: np.array = np.zeros(shape=(n, n))
    for s in states:
        routes, current_states = possible_routes(states, current_state=s)
        routes_df = pd.DataFrame(routes, columns=current_states)
        # print("Routes", routes_df)
        # print("States", current_states)
        classes: pd.DataFrame = get_classes(routes_df, current_states)
        unique_classes: list = list(classes['class'].unique())
        unique_classes = sorted([int(x) for x in unique_classes])

        for c in unique_classes:
            class_count = len(classes[classes['class'] == c])
            probability = class_count / ((n-1)**s)
            P[s-1][(s-1) + c] = probability
            print(f"P[{s-1}][{(s-1) + c}] = {class_count} / {(n-1)**s} = {probability}")
    
    #*Verify that each row sums to 1
    for i in range(n):
        row_sum = np.sum(P[i])
        if row_sum != 1:
            print("Warning, row {i}'s : {P[i]} entries do not sum 1")

    return P

def t_ij_Matrix(P_matrix: np.array) -> np.array:
    """
    The main idea is to calculate all the states t_{i,j} but keeping fixed j and then iterating over i, since 
    no matter for what i we need to calculate t_{i,j}, the recurrence formula requires j fixed and all the posible
    i's. For example,
    t_0,0 = 0
    t_1,0 = 1 + P_1,0 . t_0,0 + P_1,1 . t_1,0 + P_1,2 . t_2,0  <->  (1 - P_1,1) . t_1,0    -    P_1,2 . t_2,0 = 1
    t_2,0 = 1 + P_2,0 . t_0,0 + P_2,1 . t_1,0 + P_2,2 . t_2,0  <->      - P_2,1 . t_1,0 + (1 - P_2,2) . t_2,0 = 1
    notice that to calculate t_{1, 0} or t_{2, 0}, we require the value of the other t_{k, 0} (since we know
    t_{0, 0} = 0). Generalizing, we would only require an (n-1)x(n-1) system of equations to calculate each
    t_{i, j}.
    """
    t_ij_M = []
    rows = P_matrix.shape[0]

    for j in range(0,rows):
        valid_states = [i for i in range(rows) if i != j]

        n = rows - 1
        submatrix = np.zeros((n, n), dtype=float)
        ones = np.ones(n, dtype=float)

        for r, i in enumerate(valid_states):
            for c, d in enumerate(valid_states):
                if i == d:
                    submatrix[r, c] = 1.0 - P_matrix[i, i] # Notice that we can always factor the ti,j that we want to calculate. See in the example that t_1,0 is at the left and right side of the equation.
                else:
                    submatrix[r, c] = - P_matrix[i, d] # Else, the matrix entrance will be the probability taken to the left side of the equation
            print(submatrix[r])

        print(submatrix)
        sol_vector = np.linalg.solve(submatrix, ones)
        t = np.zeros(rows, dtype=float)
        for k, s in enumerate(valid_states):
            t[s] = sol_vector[k]
        t[j] = 0.0
        t_ij_M.append(t)

    return np.array(t_ij_M)

def calculate_t_jn(n: int, P: np.array) -> np.array:
    valid_states = [i for i in range(n-1)]
    submatrix = np.zeros((n-1, n-1), dtype=float)
    ones = np.ones(n-1, dtype=float)

    for r, i in enumerate(valid_states):
        for c, d in enumerate(valid_states):
            if i == d:
                submatrix[r, c] = 1.0 - P[i, i] # Notice that we can always factor the ti,j that we want to calculate. See in the example that t_1,0 is at the left and right side of the equation.
            else:
                submatrix[r, c] = - P[i, d] # Else, the matrix entrance will be the probability taken to the left side of the equation

    print(submatrix)
    sol_vector = np.linalg.solve(submatrix, ones)
    print(sol_vector)
    return sol_vector

if __name__ == "__main__":
    routes, current_states = possible_routes([n for n in range(1, 6)], 1)
    df = pd.DataFrame(routes, columns=[i for i in current_states])
    classes = get_classes(df, current_states)
    # print_classes_summary(classes)
    A = np.array([[-1, 1, 0, 0],
               [0, -15/16, 9/16, 6/16],
               [0, 0, -28/32, 19/32],
               [0, 0, 0, -81/256]])
    b = np.array([-1, -1, -1, -1])
    print(solve_system(A, b))
    P: np.array = build_transition_matrix(5)
    print(P)
    ts: np.array = calculate_t_jn(5, P)
    print(f"t_{1},{5} = {ts[0]}")

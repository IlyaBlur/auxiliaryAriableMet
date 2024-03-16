import numpy as np


def input_data():
    num_variables = int(input("Enter the number of decision variables: "))
    num_constraints = int(input("Enter the number of constraints: "))

    print("\nEnter the coefficients of the objective function:")
    objective_coeffs = np.array(list(map(float, input().split())))

    print("\nEnter the coefficients of constraints and the right-hand side values (RHS):")
    constraint_coeffs = []
    for _ in range(num_constraints):
        constraint_coeffs.append(list(map(float, input().split())))
    constraint_coeffs = np.array(constraint_coeffs)

    return num_variables, num_constraints, objective_coeffs, constraint_coeffs


def artificial_basis_method(num_variables, num_constraints, objective_coeffs, constraint_coeffs):
    num_artificial_variables = num_constraints
    total_variables = num_variables + num_artificial_variables

    objective_function = np.zeros(total_variables)
    objective_function[:num_variables] = -objective_coeffs

    constraint_matrix = np.eye(num_constraints, total_variables)
    constraint_matrix[:, :num_variables] = constraint_coeffs[:, :-1]

    B_inv = np.eye(num_constraints)
    c_B = objective_function[num_variables:total_variables]

    x_B = constraint_coeffs[:, -1]
    x = np.zeros(total_variables)
    x[num_variables:] = x_B

    z = np.dot(c_B, np.dot(B_inv, constraint_matrix)) - objective_function
    z_0 = np.dot(c_B, np.dot(B_inv, x_B))

    iteration_count = 0

    while np.any(z < 0):
        entering_variable = np.argmin(z)
        if np.all(constraint_matrix[:, entering_variable] <= 0):
            return None

        leaving_variable = np.where(constraint_matrix[:, entering_variable] > 0,
                                    x_B / constraint_matrix[:, entering_variable], np.inf).argmin()

        pivot_row = constraint_matrix[leaving_variable]
        scale_by = pivot_row[entering_variable]
        constraint_matrix[leaving_variable] /= scale_by
        x_B[leaving_variable] /= scale_by

        for i in range(num_constraints):
            if i == leaving_variable:
                continue
            scale_by = constraint_matrix[i, entering_variable]
            constraint_matrix[i] -= scale_by * constraint_matrix[leaving_variable]
            x_B[i] -= scale_by * x_B[leaving_variable]

        c_B[leaving_variable], objective_function[entering_variable] = objective_function[entering_variable], c_B[
            leaving_variable]

        z = np.dot(c_B, constraint_matrix) - objective_function
        z_0 = np.dot(c_B, x_B)

        iteration_count += 1

    x_B = np.linalg.lstsq(constraint_matrix[:, :num_variables], x_B, rcond=None)[0]
    x[:num_variables] = x_B
    return x, z_0, iteration_count


def main():
    num_variables, num_constraints, objective_coeffs, constraint_coeffs = input_data()
    result = artificial_basis_method(num_variables, num_constraints, objective_coeffs, constraint_coeffs)

    if result is None:
        print("The linear programming problem is unbounded.")
    else:
        solution, objective_value, iterations = result
        print(f"\nSolution found in {iterations} iterations:")
        print(f"Objective function value: {objective_value}")

        for i, x_value in enumerate(solution[:num_variables]):
            print(f"x_{i + 1} = {x_value}")


if __name__ == "__main__":
    main()
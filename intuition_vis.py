import numpy as np
import matplotlib.pyplot as plt

def vis1():
    # Define the function and its der   ivative
    def f(x):
        return x**2 + 2

    def f_prime(x):
        return 2*x  # Derivative of f(x) = x^2 + 2

    # Generate x values
    x = np.linspace(-2, 2, 400)  # Range from -2 to 2 with 400 points
    y = f(x)

    # Points where we draw tangent lines
    tangent_points = [-1.5, 0, 0.5]

    # Generate plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=r'$f(x) = x^2 + 2$', color='b')

    # Plot tangent lines and dots
    for x0 in tangent_points:
        slope = f_prime(x0)  # f'(x0)
        y0 = f(x0)  # f(x0)
        tangent_line = slope * (x - x0) + y0  # h(x) = f'(x0) * (x - x0) + f(x0)
        plt.plot(x, tangent_line, linestyle='dashed', label=f'Tangent at x={x0:.1f}')
        
        # Add dot at (x0, f(x0))
        plt.scatter(x0, y0, color='red', s=40, zorder=3)  # Red dot, size 80
        
        # Label the tangent line at (x0, h(x0))
        plt.text(x0, y0, f'h({x0})', fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    # Adjust x and y axis range
    plt.xlim(-2.5, 2.5)  # Custom x-axis range
    plt.ylim(0, 6)  # Custom y-axis range

    # Add reference lines
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Labels and title
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Graph of $f(x) = x^2 + 2$ with Tangent Lines and Points')
    plt.legend()

    # Show the plot
    plt.show()

def vis2():
    # Define the function and its derivative
    def f(x):
        return (2*x**4) - (3*x**3 ) + (5*x**2) - (7*x) + 4

    def f_prime(x):
        return (8*x**3) - (9*x**2) + (10*x) - 7  # Derivative of f(x)

    # Generate x values
    x = np.linspace(-5, 5, 400)  # Range from -5 to 5 with 400 points
    y = f(x)

    # Points where we draw tangent lines
    tangent_points = [-0.4, 0.1, 0.6, 1.1, 0.85]

    # Create plot with enough space for the table
    fig, ax = plt.subplots(figsize=(8, 6))  
    ax.plot(x, y, label=r'$f(x) = 2x^4 - 3x^3 + 5x^2 - 7x + 4$', color='b')

    # Prepare data for the table
    table_data = []
    for i, x0 in enumerate(tangent_points):
        slope = f_prime(x0)  # f'(x0)
        y0 = f(x0)  # f(x0)
        tangent_line = slope * (x - x0) + y0  # h(x) = f'(x0) * (x - x0) + f(x0)
        
        ax.plot(x, tangent_line, linestyle='dashed', label=f'Tangent at $x_{i}={x0}$')
        
        # Add dot at (x0, f(x0))
        ax.scatter(x0, y0, color='red', s=40, zorder=3)  # Red dot, size 40
        
        # Label the tangent line at (x0, h(x0))
        ax.text(x0, y0, f'$h(x_{i})$', fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, boxstyle='round,pad=0.3'))
        
        # Store data for table
        table_data.append([f"$x_{i}$", f"{x0:.2f}", f"{slope:.2f}"])

    # Adjust x and y axis range
    ax.set_xlim(-1, 1.5)  # Custom x-axis range
    ax.set_ylim(-1, 10)  # Custom y-axis range

    # Add reference lines
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Finding $x_{min}$ of $f(x) = 2x^4 - 3x^3 + 5x^2 - 7x + 4$')
    ax.legend()

    # Add table to the right side of the plot
    table = plt.table(cellText=table_data, 
                    colLabels=["Iteration", "x", "f'(x)"],
                    cellLoc='center', loc='right', bbox=[1.2, 0.2, 0.3, 0.5])

    # Show the plot
    plt.show()

def vis3():

    # Define the function and its derivative
    def f(x):
        return (2*x**4) - (3*x**3 ) + (5*x**2) - (7*x) + 4

    def f_prime(x):
        return (8*x**3) - (9*x**2) + (10*x) - 7  # Derivative of f(x)

    # Gradient descent update rule
    def gradient_descent_update(x, t):
        return x - t * f_prime(x)

    # Parameters
    t = 0.075  # Learning rate (step size)
    x0 = -0.4  # Initial guess
    num_iterations = 4  # Number of iterations
    tangent_points = [x0]  # Store the sequence of x values

    # Perform gradient descent
    for _ in range(num_iterations):
        x_new = gradient_descent_update(tangent_points[-1], t)
        tangent_points.append(x_new)


    # Generate x values
    x = np.linspace(-5, 5, 400)  # Range from -5 to 5 with 400 points
    y = f(x)

    # Create plot with enough space for the table
    fig, ax = plt.subplots(figsize=(8, 6))  
    ax.plot(x, y, label=r'$f(x) = 2x^4 - 3x^3 + 5x^2 - 7x + 4$', color='b')

    # Prepare data for the table
    table_data = []
    for i, x0 in enumerate(tangent_points):
        slope = f_prime(x0)  # f'(x0)
        y0 = f(x0)  # f(x0)
        tangent_line = slope * (x - x0) + y0  # h(x) = f'(x0) * (x - x0) + f(x0)
        
        ax.plot(x, tangent_line, linestyle='dashed', label=f'Tangent at $x_{i}={x0:.4f}$')
        
        # Add dot at (x0, f(x0))
        ax.scatter(x0, y0, color='red', s=40, zorder=3)  # Red dot, size 40
        
        # Label the tangent line at (x0, h(x0))
        ax.text(x0, y0, f'$h(x_{i})$', fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, boxstyle='round,pad=0.3'))
        
        # Store data for table
        table_data.append([f"$x_{i}$", f"{x0:.4f}", f"{slope:.4f}"])

    # Adjust x and y axis range
    ax.set_xlim(-1, 1.5)  # Custom x-axis range
    ax.set_ylim(-1, 10)  # Custom y-axis range

    # Add reference lines
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Finding $x_{min}$ of $f(x)$ using the new update rule ($t = 0.075$)')
    ax.legend()

    # Add table to the right side of the plot
    table = plt.table(cellText=table_data, 
                    colLabels=["Iteration", "x", "f'(x)"],
                    cellLoc='center', loc='right', bbox=[1.2, 0.2, 0.3, 0.5])

    # Show the plot
    plt.show()

def vis4():
    # Define the function
    def f(x):
        return np.where(x > 0, (2*x**4) - (3*x**3) + (5*x**2) - (7*x) + 4, -15*x + 4)

    # Generate x values
    x = np.linspace(-5, 5, 400)
    y = f(x)

    # Point where function is not differentiable
    x0 = 0
    y0 = f(x0)

    # Different subgradients at x = 0
    slopes = [-15, -7, -10]  # Left derivative = -15, Right derivative ~ -7, Extra slope for illustration

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, label=r'$f(x)$', color='b')

    # Plot multiple 'tangent' lines at x=0
    for slope in slopes:
        tangent_line = slope * (x - x0) + y0
        ax.plot(x, tangent_line, linestyle='dashed', label=f'Slope = {slope}')

    # Add dot at the kink (x0, f(x0))
    ax.scatter(x0, y0, color='red', s=80, zorder=3)

    # Adjust x and y axis range
    ax.set_xlim(-1, 1.5)
    ax.set_ylim(-1, 10)

    # Add reference lines
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f"'Tangent lines' at x = 0 for $f(x)$")
    ax.legend()

    plt.show()

def vis5():

    # Define the function and its derivative
    def f(x):
        return np.where(x > 0, (2*x**4) - (3*x**3) + (5*x**2) - (7*x) + 4, -15*x + 4)

    def f_prime(x):
        if x == 0:
            m = -10
            return m
        else:
            return np.where(x > 0, (8*x**3) - (9*x**2) + (10*x) - 7, -15)  # Derivative of f(x) when x is not 0

    # Gradient descent update rule
    def gradient_descent_update(x, t):
        return x - t * f_prime(x)

    # Parameters
    t = 0.075  # Learning rate (step size)
    x0 = 0  # Initial guess
    num_iterations = 3  # Number of iterations
    tangent_points = [x0]  # Store the sequence of x values

    # Perform gradient descent
    for _ in range(num_iterations):
        x_new = gradient_descent_update(tangent_points[-1], t)
        tangent_points.append(x_new)


    # Generate x values
    x = np.linspace(-5, 5, 400)  # Range from -5 to 5 with 400 points
    y = f(x)

    # Create plot with enough space for the table
    fig, ax = plt.subplots(figsize=(8, 6))  
    ax.plot(x, y, label=r'$f(x) = 2x^4 - 3x^3 + 5x^2 - 7x + 4$', color='b')

    # Prepare data for the table
    table_data = []
    for i, x0 in enumerate(tangent_points):
        slope = f_prime(x0)  # f'(x0)
        y0 = f(x0)  # f(x0)
        tangent_line = slope * (x - x0) + y0  # h(x) = f'(x0) * (x - x0) + f(x0)
        
        ax.plot(x, tangent_line, linestyle='dashed', label=f'Tangent at $x_{i}={x0:.4f}$')
        
        # Add dot at (x0, f(x0))
        ax.scatter(x0, y0, color='red', s=40, zorder=3)  # Red dot, size 40
        
        # Label the tangent line at (x0, h(x0))
        ax.text(x0, y0, f'$h(x_{i})$', fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, boxstyle='round,pad=0.3'))
        
        # Store data for table
        table_data.append([f"$x_{i}$", f"{x0:.4f}", f"{slope:.4f}"])

    # Adjust x and y axis range
    ax.set_xlim(-1, 1.5)  # Custom x-axis range
    ax.set_ylim(-1, 10)  # Custom y-axis range

    # Add reference lines
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title("Finding $x_{min}$ of the new $f(x)$ where $f'(0) = -10$")
    ax.legend()

    # Add table to the right side of the plot
    table = plt.table(cellText=table_data, 
                    colLabels=["Iteration", "x", "f'(x)"],
                    cellLoc='center', loc='right', bbox=[1.2, 0.2, 0.3, 0.5])

    # Show the plot
    plt.show() 

def vis6():

    # Define the function and its derivative
    def f(x):
        return (2*x**4) - (3*x**3 ) + (5*x**2) - (7*x) + 4

    def f_prime(x):
        return (8*x**3) - (9*x**2) + (10*x) - 7  # Derivative of f(x)

    # Gradient descent update rule
    def gradient_descent_update(x, t):
        return x - t * f_prime(x)

    # Parameters
    t = 0.14  # Learning rate (step size)
    x0 = -0.4  # Initial guess
    num_iterations = 6  # Number of iterations
    tangent_points = [x0]  # Store the sequence of x values

    # Perform gradient descent
    for _ in range(num_iterations):
        x_new = gradient_descent_update(tangent_points[-1], t)
        tangent_points.append(x_new)


    # Generate x values
    x = np.linspace(-5, 5, 400)  # Range from -5 to 5 with 400 points
    y = f(x)

    # Create plot with enough space for the table
    fig, ax = plt.subplots(figsize=(8, 6))  
    ax.plot(x, y, label=r'$f(x) = 2x^4 - 3x^3 + 5x^2 - 7x + 4$', color='b')

    # Prepare data for the table
    table_data = []
    for i, x0 in enumerate(tangent_points):
        slope = f_prime(x0)  # f'(x0)
        y0 = f(x0)  # f(x0)
        tangent_line = slope * (x - x0) + y0  # h(x) = f'(x0) * (x - x0) + f(x0)
        
        ax.plot(x, tangent_line, linestyle='dashed', label=f'Tangent at $x_{i}={x0:.4f}$')
        
        # Add dot at (x0, f(x0))
        ax.scatter(x0, y0, color='red', s=40, zorder=3)  # Red dot, size 40
        
        # Label the tangent line at (x0, h(x0))
        ax.text(x0, y0, f'$h(x_{i})$', fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, boxstyle='round,pad=0.3'))
        
        # Store data for table
        table_data.append([f"$x_{i}$", f"{x0:.4f}", f"{slope:.4f}"])

    # Adjust x and y axis range
    ax.set_xlim(-1, 1.5)  # Custom x-axis range
    ax.set_ylim(-1, 10)  # Custom y-axis range

    # Add reference lines
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Finding $x_{min}$ of $f(x)$ using the new update rule ($t = 0.14$)')
    ax.legend()

    # Add table to the right side of the plot
    table = plt.table(cellText=table_data, 
                    colLabels=["Iteration", "x", "f'(x)"],
                    cellLoc='center', loc='right', bbox=[1.2, 0.2, 0.3, 0.5])

    # Show the plot
    plt.show()
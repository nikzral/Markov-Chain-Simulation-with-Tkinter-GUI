import numpy as np
from numba import jit
from tqdm import tqdm
from tqdm import trange
import tkinter as tk
from tkinter import Canvas

# Define the states
states = ['Burger', 'Pizza', 'Hotdogs']

# Transition probability matrix
probability_matrix = np.array([[0.1, 0.6, 0.3],
                               [0.4, 0.2, 0.4],
                               [0.3, 0.3, 0.4]])

# ASCII visualization of the Markov chain
def print_ascii_chain(states, matrix):
    print("\nMarkov Chain State Transitions:\n")
    print(" " * 7 + " ".join([f"{s:^7}" for s in states]))
    for i, from_state in enumerate(states):
        transitions = " ".join([f"{matrix[i, j]:^7.2f}" for j in range(len(states))])
        print(f"{from_state:^7}" + transitions)

# Display the probability matrix
def print_probability_matrix(matrix):
    print("\nTransition Probability Matrix:\n")
    for row in matrix:
        for prob in row:
            print(f"{prob:7.4f}", end="")
        print()

# JIT-optimized custom choice function
@jit(nopython=True)
def custom_choice(probabilities):
    x = np.random.random()
    cumulative = 0
    for i, p in enumerate(probabilities):
        cumulative += p
        if x < cumulative:
            return i
    return len(probabilities) - 1

# JIT-compiled function to simulate Markov chain transitions
@jit(nopython=True)
def simulate_markov_chain(matrix, simulations, n_states):
    state = np.random.randint(n_states)
    counts = np.zeros((simulations, n_states), dtype=np.int32)
    
    for i in range(simulations):
        counts[i, state] += 1
        random_value = np.random.random()
        cumulative_probability = 0.0
        for j in range(n_states):
            cumulative_probability += matrix[state, j]
            if random_value < cumulative_probability:
                state = j
                break

    return counts.sum(axis=0) / simulations


class MarkovChainGUI(tk.Tk):
    def __init__(self, states, matrix, chunk_size=100):
        super().__init__()
        self.states = states
        self.matrix = matrix
        self.chunk_size = chunk_size
        self.simulated_steps = 0
        self.simulation_results = np.zeros(len(states))
        self.title('Markov Chain Visualization')
        self.geometry('600x400')
        self.canvas = tk.Canvas(self, bg='white', width=600, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.draw_states()

    def draw_states(self):
        circle_distance = 150
        radius = 50
        self.nodes = []
        
        for i, state in enumerate(self.states):
            x = circle_distance * (i + 1)
            y = 200
            self.nodes.append((x, y))
            self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill='lightblue')
            self.canvas.create_text(x, y, text=state)
        self.update()  # Initial update to set border widths and probabilities

    def update(self):
        max_border_width = 10  # Define a maximum border width
        for i, (prob, (x, y)) in enumerate(zip(self.simulation_results, self.nodes)):
            radius = 50
            border_width = max_border_width * prob  # Calculate border width based on probability
            self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, width=border_width)
            self.canvas.create_text(x, y, text=self.states[i])
            self.canvas.create_text(x, y+70, text=f"{prob:.4f}", fill='black')

    def run_simulation(self):
        for _ in trange(self.simulated_steps, desc='Simulating', unit='sim'):
            partial_results = simulate_markov_chain(self.matrix, self.chunk_size, len(self.states))
            self.simulation_results += partial_results * self.chunk_size
            self.canvas.delete("all")
            self.draw_states()
            self.simulation_results /= (self.chunk_size * (_ + 1))
            self.simulated_steps += 1
            self.update()
            self.canvas.update_idletasks()
            self.update_idletasks()

    def start_simulation(self, total_simulations):
        self.simulated_steps = total_simulations // self.chunk_size
        self.run_simulation()

if __name__ == "__main__":
    states = ['Burger', 'Pizza', 'Hotdogs']
    probability_matrix = np.array([[0.1, 0.6, 0.3], [0.4, 0.2, 0.4], [0.3, 0.3, 0.4]])
    
    try:
        simulations = int(input("Enter the number of simulations to perform: "))
    except ValueError:
        print("Please enter a valid integer for the number of simulations.")
        exit(1)
    
    app = MarkovChainGUI(states, probability_matrix, chunk_size=5)
    app.after(100, app.start_simulation, simulations)  # Delay start to allow window to render
    app.mainloop()
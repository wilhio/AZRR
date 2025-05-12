import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np

"""
AZRR with Monte Carlo Integration
"""
class MonteCarloDataset:
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        
    def generate_integration_task(self):
        # Generate a random function to integrate
        a = np.random.uniform(-5, 5)
        b = np.random.uniform(-5, 5)
        c = np.random.uniform(-2, 2)
        return lambda x: a * np.sin(b * x) + c * x**2
    
    def generate_tasks(self, num_tasks):
        tasks = []
        for _ in range(num_tasks):
            func = self.generate_integration_task()
            x_samples = np.random.uniform(-5, 5, self.num_samples)
            y_samples = func(x_samples)
            tasks.append((func, x_samples, y_samples))
        return tasks

# Simplified transformer model (replace with a real LLM in practice)
class SimpleLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)  # Placeholder for a real transformer
    
    def forward(self, x):
        return self.layer(x)

# AZR class for self-improvement
class AZR:
    def __init__(self, model):
        self.model = model
        self.dataset = MonteCarloDataset()
    
    def generate_tasks(self, num_tasks):
        return self.dataset.generate_tasks(num_tasks)
    
    def solve_tasks(self, tasks):
        solutions = []
        for func, x_samples, y_samples in tasks:
            # Monte Carlo integration: area = (b-a) * mean(f(x))
            area = 10 * np.mean(y_samples)  # 10 is the width of our integration interval [-5,5]
            solutions.append(area)
        return solutions
    
    def evaluate_solutions(self, tasks, solutions):
        rewards = []
        errors = []
        for (func, _, _), sol in zip(tasks, solutions):
            # Calculate true integral using numerical integration
            x = np.linspace(-5, 5, 1000)
            y = func(x)
            true_integral = np.trapezoid(y, x)  # Using trapezoid instead of trapz
            
            # Calculate relative error
            rel_error = abs(sol - true_integral) / (abs(true_integral) + 1e-10)
            errors.append(rel_error)
            
            # Reward based on relative error
            reward = 1.0 if rel_error < 0.1 else max(0, 1 - rel_error)
            rewards.append(reward)
        
        # Calculate accuracy metrics
        mean_error = np.mean(errors)
        accuracy = np.mean(rewards)
        return rewards, mean_error, accuracy
    
    def update_model(self, tasks, solutions, rewards):
        # Placeholder for RL update (e.g., policy gradient)
        # In practice, use an optimizer to adjust self.model parameters
        pass
    
    def train(self, iterations):
        total_accuracy = 0
        total_error = 0
        for i in range(iterations):
            tasks = self.generate_tasks(5)
            solutions = self.solve_tasks(tasks)
            rewards, mean_error, accuracy = self.evaluate_solutions(tasks, solutions)
            self.update_model(tasks, solutions, rewards)
            
            total_accuracy += accuracy
            total_error += mean_error
            
            print(f"Step {i+1}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Mean Relative Error: {mean_error:.4f}")
            print(f"  Estimated vs True Integral Examples:")
            for j, (func, _, _) in enumerate(tasks[:2]):  # Show first 2 examples
                x = np.linspace(-5, 5, 1000)
                y = func(x)
                true_integral = np.trapezoid(y, x)
                print(f"    Task {j+1}:")
                print(f"      Estimated: {solutions[j]:.4f}")
                print(f"      True: {true_integral:.4f}")
        
        avg_accuracy = total_accuracy / iterations
        avg_error = total_error / iterations
        print(f"\nTraining Summary:")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Relative Error: {avg_error:.4f}")
        return avg_accuracy

# AZRR framework with parent-child interaction
class AZRR:
    def __init__(self):
        base_model = SimpleLLM()
        self.parent = AZR(deepcopy(base_model))
        self.child = AZR(deepcopy(base_model))
        self.child_performance = 0.0
        
        # Pre-train parent to make it competent
        print("Pre-training parent...")
        self.parent.train(10)  # Run AZR loop for 10 iterations
    
    def parent_generate_tasks_for_child(self, num_tasks):
        # Generate Monte Carlo tasks with adjusted difficulty
        difficulty = min(10, max(1, int(5 * (1 - self.child_performance) + 1)))
        dataset = MonteCarloDataset(num_samples=difficulty * 100)  # More samples for harder tasks
        return dataset.generate_tasks(num_tasks)
    
    def child_train_step(self):
        # Child generates its own tasks
        child_tasks = self.child.generate_tasks(3)
        # Parent generates tasks for child
        parent_tasks = self.parent_generate_tasks_for_child(3)
        all_tasks = child_tasks + parent_tasks
        
        solutions = self.child.solve_tasks(all_tasks)
        rewards, mean_error, accuracy = self.child.evaluate_solutions(all_tasks, solutions)
        self.child.update_model(all_tasks, solutions, rewards)
        
        # Update child performance
        self.child_performance = accuracy
        return self.child_performance
    
    def parent_train_step(self):
        # Parent continues self-improvement
        return self.parent.train(1)
    
    def run(self, iterations):
        print("Starting AZRR training...")
        for i in range(iterations):
            print(f"\nIteration {i+1}:")
            child_perf = self.child_train_step()
            parent_perf = self.parent_train_step()
            print(f"Child Accuracy: {child_perf:.4f}")
            print(f"Parent Accuracy: {parent_perf:.4f}")

# Run the AZRR framework
if __name__ == "__main__":
    azrr = AZRR()
    azrr.run(5)
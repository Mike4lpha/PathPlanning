#! /usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to load the maze from a bitmap image
def load_maze_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 0 = free, 255 = obstacle
    _, binary_maze = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY_INV)
    return binary_maze

# Function to visualize the maze with the computed path
def visualize_path(maze, path, start=None, goal=None):
    maze_copy = np.copy(maze)

    for point in path:
        maze_copy[point] = 2  

    if start:
        maze_copy[start] = 4
    if goal:
        maze_copy[goal] = 3  

    # Display the maze with the path
    plt.figure(figsize=(10, 10))
    plt.imshow(maze_copy, cmap='binary')
    plt.title("Maze with Path")
    plt.axis('off')
    plt.show()

# Line Following Algorithm
def line_following(maze, start, goal):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    current = start
    path = [current]

    while current != goal:
        for dx, dy in directions:
            next_ = (current[0] + dx, current[1] + dy)
            if 0 <= next_[0] < maze.shape[0] and 0 <= next_[1] < maze.shape[1] and maze[next_] == 0:
                current = next_
                path.append(current)
                break
        else:
            print("No valid moves. Line following failed.")
            return []

    return path

# BFS Algorithm
from collections import deque
def bfs(maze, start, goal):
    rows, cols = maze.shape
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    queue = deque([start])
    came_from = {start: None}  # Keep track of the path

    while queue:
        current = queue.popleft()

        if current == goal:
            # Reconstruct the path
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                    maze[neighbor] == 0 and neighbor not in came_from):
                queue.append(neighbor)
                came_from[neighbor] = current

    print("Goal not reachable.")
    return []

if __name__ == "__main__":
    image_path = "map.bmp"  # Replace with your maze image path
    maze = load_maze_from_image(image_path)
    
    start = (340, 220)  
    goal = (189, 310)  

    path = bfs(maze, start, goal)  

    print("Path:", path)
    visualize_path(maze, path, start=start, goal=goal)

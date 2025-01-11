#! /usr/bin/env python3
import numpy as np
import cv2
from heapq import heappop, heappush
import matplotlib.pyplot as plt

def load_maze(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_maze = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY_INV)
    return binary_maze, image

def heuristic(a, b):
    # Manhattan distance heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(maze, start, goal):
    rows, cols = maze.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while open_set:
        _, current = heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor] == 0:
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
    
    return [] 

def get_corner_points(path):
    corners = [path[0]]
    for i in range(1, len(path) - 1):
        prev = path[i - 1]
        curr = path[i]
        next_ = path[i + 1]
        if (curr[0] - prev[0], curr[1] - prev[1]) != (next_[0] - curr[0], next_[1] - curr[1]):
            corners.append(curr)
    corners.append(path[-1])
    return corners

def visualize_path(maze_image, path, corners):
    path = np.array(path)
    corners = np.array(corners)

    color_maze = cv2.cvtColor(maze_image, cv2.COLOR_GRAY2BGR)

    for point in path:
        color_maze[point[0], point[1]] = (255, 0, 0) 

    for point in corners:
        cv2.circle(color_maze, (point[1], point[0]), 2, (0, 255, 0), -1)  

    # Display the maze with the path
    plt.figure(figsize=(10, 10))
    plt.title("Path Visualization")
    plt.imshow(color_maze[..., ::-1])
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    maze, maze_image = load_maze("map.bmp")
    start = (288, 429)  
    goal = (278, 70) 
    
    path = a_star(maze, start, goal)
    if path:
        corners = get_corner_points(path)
        print("Corners of the path:", corners)
        visualize_path(maze_image, path, corners)
    else:
        print("No path found.")

#! /usr/bin/env python3
import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt

def load_maze(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary_maze = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)

    plt.figure(figsize=(10, 10))
    plt.title("Binary Maze")
    plt.imshow(binary_maze, cmap='gray')
    plt.axis("off")
    plt.show()
    
    return binary_maze, image

def dijkstra(maze, start, goal):
    rows, cols = maze.shape
    distances = np.full((rows, cols), np.inf)  # Distance matrix initialized to infinity
    distances[start] = 0
    previous_nodes = {start: None}  # Dictionary to store the previous node for path reconstruction
    
    pq = [(0, start)]  # Priority queue for exploring nodes (distance, node)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Directions: right, down, left, up
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node == goal:
            # Reconstruct path from goal to start
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = previous_nodes.get(current_node)
            return path[::-1]  # Return reversed path (start -> goal)
        
        for dx, dy in directions:
            neighbor = (current_node[0] + dx, current_node[1] + dy)
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor] == 1:
                new_distance = current_distance + 1  # Cost of moving to a neighboring node is 1
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(pq, (new_distance, neighbor))
    
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
    if not path:
        print("No path found to visualize.")
        return

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
    
    start = (340, 220)  
    goal = (189, 310) 
    
    print(f"Maze value at start {start}: {maze[start]}")
    print(f"Maze value at goal {goal}: {maze[goal]}")
    
    path = dijkstra(maze, start, goal)
    
    # if path:
    #     print(f"Path found: {path}")
    # else:
    #     print("No path found.")
    
    if path:
        corners = get_corner_points(path)
        print("Corners of the path:", corners)
        visualize_path(maze_image, path, corners)
    else:
        print("No path found.")

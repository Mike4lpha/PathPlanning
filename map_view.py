#! /usr/bin/env python3
import cv2
import matplotlib.pyplot as plt

def view_maze(image_path):

    maze_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(10, 10))
    plt.title("Maze")
    plt.imshow(maze_image, cmap="gray")
    plt.axis("on")
    plt.show()

if __name__ == "__main__":
    image_path = "map.bmp"  # Replace with your maze image path
    view_maze(image_path)

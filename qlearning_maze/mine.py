from Maze import Maze
import imageio
import numpy as np

# maze = Maze(maze_size=(10, 10))
# img = maze.draw_current_maze()
# imageio.imwrite('aaa.jpg', img)

# imageio.imwrite('maze_01.jpg', Maze(from_file='./test_world/maze_01.txt').draw_current_maze())
# imageio.imwrite('maze_02.jpg', Maze(from_file='./test_world/maze_02.txt').draw_current_maze())
# imageio.imwrite('maze_03.jpg', Maze(from_file='./test_world/maze_03.txt').draw_current_maze())
# imageio.imwrite('maze_04.jpg', Maze(from_file='./test_world/maze_04.txt').draw_current_maze())

# maze_data = np.genfromtxt('./test_world/maze_01.txt', delimiter=',', dtype=np.uint16)
# print(maze_data.shape)

# print(Maze(maze_size=(10,10)))

from Robot import Robot

maze6 = Maze(from_file='./test_world/maze_01.txt')
robot = Robot(maze6) # 记得将 maze 变量修改为你创建迷宫的变量名
robot.set_status(learning=True,testing=False)
print(robot.update())
import Robot
import Maze
import numpy as np
import imageio


# g = Maze.Maze(maze_size=(20,20), trap_number=5)
# g.maze_to_file('test_map.txt', 'test_dest.txt', 'test_trap.txt')
# imageio.imwrite('test_image.jpg', g.get_raw_maze_img())
# # g.dump_maze()

# g1 = Maze.Maze(from_file='test_map.txt')
# g1.set_dest_and_traps('test_dest.txt', 'test_trap.txt')
# print(g1)

file_prefix = './result/20_500_0.9_0.7_0.9_10'

mapfile = file_prefix+'_map.txt'
destfile = file_prefix+'_dest.txt'
trapfile = file_prefix+'_traps.txt'

## 可选的参数：
epoch = 20

epsilon0 = 0.9
alpha = 0.7
gamma = 0.9

# maze_size = (10,10)
# trap_number = 5

print(mapfile, destfile, trapfile)
g = Maze.Maze(from_file=mapfile)
g.set_dest_and_traps(dest_file=destfile, traps_file=trapfile)
print(g)
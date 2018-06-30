from grid_world import GridWorld

gw = GridWorld(num_rows=4, num_columns=4)
gw.current_state = 14
print (gw.step('RIGHT'))

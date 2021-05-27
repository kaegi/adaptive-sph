# importing package
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.style.use('ggplot')
# plt.style.use('dufte')

adaptivity = []
density_solver = []
div_solver = []
level_estimation = []
neighborhood = []
simulation_step = []

# unifrom
adaptivity.append(0)
density_solver.append(10.585776)
div_solver.append(7.657546)
level_estimation.append(0)
neighborhood.append(16.361093)
simulation_step.append(37.418878)



# adaptive
adaptivity.append(0.48174900000000004)
density_solver.append(1.6296920000000001)
div_solver.append(1.574896)
level_estimation.append(0.923118)
neighborhood.append(3.33201)
simulation_step.append(8.624171)



adaptivity = np.array(adaptivity)
density_solver = np.array(density_solver)
div_solver = np.array(div_solver)
level_estimation = np.array(level_estimation)
neighborhood = np.array(neighborhood)
simulation_step = np.array(simulation_step)
rest = simulation_step - neighborhood - level_estimation - div_solver - density_solver - adaptivity

print(adaptivity)

# create data
x = ['Uniform', 'Adaptive']
# y1 = np.array([10, 20, 10, 30])
# y2 = np.array([20, 25, 15, 25])
# y3 = np.array([12, 15, 19, 6])
# y4 = np.array([10, 29, 13, 19])

# plot bars in stack manner
top = np.array([0.] * len(x))
colors = ['r', 'g', 'b', 'y', 'm', 'c']
names = ['Merging, Sharing and Splitting', 'Level-Set Estimation', 'Constant-Density Solver', 'Divergence-Free Solver', 'Neighborhood search', 'Other']
values = [adaptivity, level_estimation, density_solver, div_solver, neighborhood, rest]
for y in values:
    top += y
for i, y in enumerate(values):
    c = colors[i]
    # print(x)
    top -= y
    plt.bar(x, y, bottom=top)
# plt.xlabel("Simulation")
plt.ylabel("Time in milliseconds")
plt.legend(names)
plt.title("Average time spent in each step of the simulation loop", pad=10)
plt.ylim(None,39)
plt.savefig('motivation-performance-chart.pdf')
# plt.subplots_adjust(top=1.1)
plt.show()
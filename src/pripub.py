import sys
import cvxpy as cp
from multiprocessing import Pipe, Process
from pabulib_parser import *
import csv
from tqdm import tqdm
import warnings


class UtilityMaximizer:
    def __init__(self, num_items, capacity, items_sizes):
        assert num_items == len(items_sizes)

        self.m = num_items  # num of items

        self.x = cp.Variable(self.m, nonneg=True)   # allocation

        # self.c = cp.Parameter(value=capacity, nonneg=True)  # capacity
        self.s = cp.Parameter(self.m, value=items_sizes/capacity, nonneg=True)  # size
        self.u = cp.Parameter(self.m)  # utility vector

        self.objective = cp.Maximize(self.u @ self.x)
        self.constraints = [
            self.x <= self.s,
            cp.sum(self.x) <= 1,
        ]
        self.problem = cp.Problem(self.objective, self.constraints)
        self.problem_solved = False

    def set_parameters(self, u_vector):
        self.u.value = u_vector

    def solve(self):
        self.problem.solve(solver='SCS')
        return self.problem.status == cp.OPTIMAL or self.problem.status == cp.OPTIMAL_INACCURATE


class AGMaximizer:
    def __init__(self, num_items, capacity, items_sizes, rho_value):
        assert num_items == len(items_sizes)

        self.m = num_items  # num of public goods
        self.rho = rho_value / 2  # learning step

        # Variables
        self.x = cp.Variable(self.m, nonneg=True)

        # Parameters
        # self.c = cp.Parameter(value=capacity, nonneg=True)  # capacity
        self.s = cp.Parameter(self.m, value=items_sizes/capacity, nonneg=True)  # size

        self.u = cp.Parameter(self.m)  # utilities
        self.z = cp.Parameter(self.m)   # previous global allocation
        self.gamma = cp.Parameter(self.m)   # Lagrange multiplier

        # Objective
        self.objective = cp.Maximize(
            cp.log(self.u @ self.x) - self.gamma @ self.x - self.rho * cp.sum_squares(self.x - self.z)
        )

        # Constraints
        self.constraints = [
            self.x <= self.s,
            cp.sum(self.x) <= 1,
        ]

        self.problem = cp.Problem(self.objective, self.constraints)
        self.problem_solved = False

    def set_parameters(self, u_vector, gamma_vector):
        self.u.value = u_vector
        self.gamma.value = gamma_vector

    def set_z_vector(self, z_vector):
        self.z.value = z_vector

    def solve(self):
        self.problem.solve(solver='SCS')
        return self.problem.status == cp.OPTIMAL or self.problem.status == cp.OPTIMAL_INACCURATE


def save_results(result_dir, results, field_names, file_name='results.csv'):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_dir + file_name, 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(results)


def save_lists(result_dir, core_z, z_list, max_x):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_dir + 'core_z.txt', 'w+') as core_z_file:
        for item in core_z:
            core_z_file.write(str(item) + ' ')

    with open(result_dir + 'z_list.txt', 'w+') as z_list_file:
        for line in z_list:
            for item in line:
                z_list_file.write(str(item) + ' ')
            z_list_file.write('\n')

    with open(result_dir + 'max_x.txt', 'w+') as max_x_file:
        for line in max_x:
            for item in line:
                max_x_file.write(str(item) + ' ')
            max_x_file.write('\n')


def project_z(z, num_items, capacity, item_sizes):
    assert num_items == len(item_sizes)

    z_hat = cp.Variable(num_items, nonneg=True)

    objective = cp.Minimize(cp.sum_squares(z_hat - z))
    constraints = [
        z_hat <= item_sizes / capacity,
        cp.sum(z_hat) <= 1,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver='SCS')
    return (problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE), z_hat.value


def create_agents_per_worker_list(num_agents, num_workers):
    agents_per_worker = [i for i in range(0, num_agents, int(np.ceil(num_agents / num_workers)))]
    num_workers -= (num_workers - len(agents_per_worker))
    agents_per_worker.append(num_agents)
    assert len(agents_per_worker) == (num_workers + 1)
    return num_workers, agents_per_worker


def run_max_worker(capacity, items_sizes, utility_matrix, pipe):
    assert len(utility_matrix.shape) == 2
    num_agents, num_items = utility_matrix.shape

    solver = UtilityMaximizer(num_items, capacity, items_sizes)

    max_allocations = []
    for i in range(num_agents):
        solver.set_parameters(utility_matrix[i, :])
        if not solver.solve():
            sys.exit('Solving primal program failed!')
        max_allocations.append(solver.x.value)

    max_allocations = np.array(max_allocations)
    pipe.send(max_allocations)


def run_admm_worker(capacity, items_sizes, rho, utility_matrix, num_steps, pipe):
    assert len(utility_matrix.shape) == 2
    num_agents, num_items = utility_matrix.shape
    z_vector = np.zeros(num_items)
    gamma_matrix = np.zeros((num_agents, num_items))

    solver = AGMaximizer(num_items, capacity, items_sizes, rho)

    for step in range(num_steps):
        optimal_allocations = []
        solver.set_z_vector(z_vector)
        for i in range(num_agents):
            solver.set_parameters(utility_matrix[i, :], gamma_matrix[i, :])
            if not solver.solve():
                sys.exit('Solving primal program failed!')

            optimal_allocations.append(solver.x.value)

        optimal_allocations = np.array(optimal_allocations)
        pipe.send(optimal_allocations)
        z_vector = pipe.recv()
        gamma_matrix += (rho * (optimal_allocations - z_vector))


def run_max(num_workers, capacity, items_sizes, utility_matrix):
    assert len(utility_matrix.shape) == 2
    num_agents, num_items = utility_matrix.shape
    assert num_items == len(items_sizes)

    num_workers, agents_per_worker = create_agents_per_worker_list(num_agents, num_workers)
    pipes = []

    print('Lunching ' + str(num_workers) + ' processes to find max allocations')
    for i in range(num_workers):
        local, remote = Pipe()
        pipes += [local]
        first_agent = agents_per_worker[i]
        last_agent = agents_per_worker[i + 1]
        Process(target=run_max_worker,
                args=(capacity, items_sizes, utility_matrix[first_agent: last_agent, :], remote)).start()

    max_allocations = np.zeros((num_agents, num_items))

    for i in range(num_workers):
        first_agent = agents_per_worker[i]
        last_agent = agents_per_worker[i + 1]
        max_x = pipes[i].recv()
        assert max_x.shape[0] == last_agent - first_agent
        max_allocations[first_agent:last_agent, :] = max_x

    return max_allocations


def run_admm(num_workers, num_steps, capacity, items_sizes, rho, utility_matrix, variance, find_core):
    assert len(utility_matrix.shape) == 2

    num_agents, num_items = utility_matrix.shape
    assert num_items == len(items_sizes)

    previous_noise = np.zeros(num_items)

    num_workers, agents_per_worker = create_agents_per_worker_list(num_agents, num_workers)
    pipes = []

    print('Lunching ' + str(num_workers) + ' processes to run ADMM for ' + str(num_steps) + ' steps')
    for i in range(num_workers):
        local, remote = Pipe()
        pipes += [local]
        first_agent = agents_per_worker[i]
        last_agent = agents_per_worker[i + 1]

        Process(target=run_admm_worker,
                args=(capacity, items_sizes, rho, utility_matrix[first_agent: last_agent, :],
                      num_steps, remote)).start()

    z_vector_bar = np.zeros(num_items)

    for _ in tqdm(range(num_steps)):
        # print('Step', step)
        z_vector = np.zeros(num_items)
        for pipe in pipes:
            optimal_x = pipe.recv()
            for i in range(len(optimal_x)):
                z_vector += optimal_x[i]

        z_vector /= num_agents
        z_vector_bar += z_vector

        # adding noise for DP
        if not find_core:
            noise = np.random.normal(0, np.sqrt(variance), num_items)
            z_vector += (noise - previous_noise)
            previous_noise = noise
            # print('variance:', variance)

        # print('capacity violation:', max(0, items_sizes @ z_vector - capacity))

        for pipe in pipes:
            pipe.send(z_vector)

    z_vector_bar += previous_noise
    z_vector_bar /= num_steps

    if not find_core:
        projected, z_vector_bar = project_z(z_vector_bar, num_items, capacity, items_sizes)
        assert projected

    return z_vector_bar


def run_scenarios(final_data_folder='final_data', final_results_folder='final_results',
                  num_runs=50, num_workers=220):
    final_data_dir = project_dir + final_data_folder
    results_dir = project_dir + final_results_folder + '/'
    field_names = ['scenario_name', 'num_agents', 'num_items',
                   'epsilon', 'delta', 'alpha', 'num_steps', 'variance', 'noise_level',
                   'min_pos_u', 'max_pos_u', 'avg_pos_u', 'var_pos_u',
                   'core_sw', 'core_min_sw_si', 'core_max_sw_si', 'core_avg_sw_si', 'core_var_sw_si',
                   'ppga_sw', 'ppga_min_sw_si', 'ppga_max_sw_si', 'ppga_avg_sw_si', 'ppga_var_sw_si',
                   'ppga_capacity_violation', 'ppga_core_z_distance',
                   'ppga_core_sw_avg', 'ppga_core_sw_var']

    results = []
    for scenario in glob.glob(final_data_dir + '/*'):
        scenario_name = re.split('/', scenario)[-1]
        print('Started ' + scenario_name)

        print('Reading saved data ...')
        num_agents, num_items, capacity, items_sizes, utility_matrix = read_saved_scenario(scenario)

        pos_u = np.sum(utility_matrix > 0, axis=1)
        min_pos_u = np.amin(pos_u)
        max_pos_u = np.amax(pos_u)
        avg_pos_u = np.average(pos_u)
        var_pos_u = np.var(pos_u)

        # Parameters
        num_steps = int(num_agents / 1000)
        epsilon = 1.5 / np.log10(num_agents)
        assert epsilon < 1
        delta = 0.3 / (num_agents**0.5)
        rho = 0.01
        alpha = 1 + (np.log10(1 / delta) / (0.5 * epsilon))
        epsilon_prime = 0.5 * epsilon / num_steps
        variance = alpha / ((num_agents ** 2) * epsilon_prime)
        noise_level = (num_steps * num_items * alpha) / (num_agents**2 * 0.5 * epsilon)

        print('Number of agents: ' + str(num_agents) + ', number of items: ' + str(num_items) +
              ', number of iterations: ' + str(num_steps) + ', epsilon: ' + str(epsilon) + ', delta: ' + str(delta) +
              ', alpha: ' + str(alpha) + ', variance: ' + str(variance) +
              ', noise_level: ' + str(noise_level))

        print('Finding a core solution ...')
        core_z = run_admm(num_workers, 2 * num_steps, capacity, items_sizes, rho,
                          utility_matrix, variance, find_core=True)
        core_sw = np.sum(core_z * utility_matrix) / num_agents
        # print('Social welfare for core: ' + str(core_sw))
        # print('Capacity violation for core: ' + str(items_sizes @ core_z - capacity))

        print('Finding max allocations ...')
        max_x = run_max(num_workers, capacity, items_sizes, utility_matrix)
        assert max_x.shape == utility_matrix.shape
        max_u = np.sum(max_x * utility_matrix, axis=1) / num_agents

        core_sw_u = np.sum(core_z * utility_matrix, axis=1)
        core_sw_si = core_sw_u / max_u
        core_min_sw_si = np.amin(core_sw_si)
        core_max_sw_si = np.amax(core_sw_si)
        core_avg_sw_si = np.average(core_sw_si)
        core_var_sw_si = np.var(core_sw_si)
        # print('Core min si: ' + str(core_min_sw_si) + ', core max si: ' + str(core_max_sw_si))

        ppga_sw = 0
        ppga_capacity_violation = 0
        ppga_core_z_distance = 0
        ppga_min_sw_si = 0
        ppga_max_sw_si = 0
        ppga_avg_sw_si = 0
        ppga_var_sw_si = 0

        z_list = []
        ppga_core_sw_list = []

        for i in range(num_runs):
            print('Running PPGA for experiment number: ' + str(i))
            z = run_admm(num_workers, num_steps, capacity, items_sizes, rho,
                         utility_matrix, variance, find_core=False)
            ppga_sw_cur = np.sum(z * utility_matrix) / num_agents
            ppga_core_sw_list.append(ppga_sw_cur / core_sw)
            ppga_sw += ppga_sw_cur
            # print('Social welfare for PPGA: ' + str(core_sw))
            # print('Capacity violation for PPGA: ' + str(items_sizes @ core_z - capacity))
            ppga_sw_u = np.sum(z * utility_matrix, axis=1)
            ppga_sw_si = ppga_sw_u / max_u
            ppga_min_sw_si += np.amin(ppga_sw_si)
            ppga_max_sw_si += np.amax(ppga_sw_si)
            ppga_avg_sw_si += np.average(ppga_sw_si)
            ppga_var_sw_si += np.var(ppga_sw_si)
            # print('PPGA min si: ' + str(ppga_min_sw_si) + ', PPGA max si: ' + str(ppga_max_sw_si))
            ppga_capacity_violation += max(0, items_sizes @ z - capacity)
            ppga_core_z_distance += abs(z - core_z).sum() / (2 * num_items)
            z_list.append(z)

        data = {'scenario_name': scenario_name, 'num_agents': num_agents, 'num_items': num_items,
                'epsilon': epsilon, 'delta': delta, 'alpha': alpha, 'num_steps': num_steps,
                'variance': variance, 'noise_level': noise_level,
                'min_pos_u': min_pos_u, 'max_pos_u': max_pos_u, 'avg_pos_u': avg_pos_u, 'var_pos_u': var_pos_u,
                'core_sw': core_sw, 'core_min_sw_si': core_min_sw_si, 'core_max_sw_si': core_max_sw_si,
                'core_avg_sw_si': core_avg_sw_si, 'core_var_sw_si': core_var_sw_si,
                'ppga_sw': ppga_sw / num_runs, 'ppga_min_sw_si': ppga_min_sw_si / num_runs,
                'ppga_max_sw_si': ppga_max_sw_si / num_runs, 'ppga_avg_sw_si': ppga_avg_sw_si / num_runs,
                'ppga_var_sw_si': ppga_var_sw_si / num_runs,
                'ppga_capacity_violation': ppga_capacity_violation / num_runs,
                'ppga_core_z_distance': ppga_core_z_distance / num_runs,
                'ppga_core_sw_avg': np.mean(ppga_core_sw_list),
                'ppga_core_sw_var': np.var(ppga_core_sw_list)}

        assert len(data) == len(field_names)
        results.append(data)
        # print(data)

        save_lists(results_dir + scenario_name + '/', core_z, z_list, max_x)
        save_results(results_dir, results, field_names)

    save_results(results_dir, results, field_names)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_scenarios(final_data_folder='final_data', final_results_folder='final_results',
                      num_runs=50, num_workers=250)

import csv
import os
import numpy as np
import re
import glob

project_dir = os.getcwd() + '/'


def process_raw_data(raw_data_folder='raw_data', final_data_folder='final_data'):
    for in_path in glob.glob(project_dir + raw_data_folder + '/*.pb'):
        file_name = re.split(r'/|\.', in_path)[-2]
        out_path = project_dir + final_data_folder + '/' + file_name
        with open(in_path, 'r', newline='', encoding='utf-8') as csvfile:
            meta = {}
            projects = {}
            votes = {}
            section = ''
            header = []
            utility_matrix = {}

            num_voters = 0
            num_projects = 0

            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                if str(row[0]).strip().lower() in ['meta', 'projects', 'votes']:
                    section = str(row[0]).strip().lower()
                    header = next(reader)
                elif section == 'meta':
                    meta[row[0]] = row[1].strip()
                elif section == 'projects':
                    projects[row[0]] = {}
                    projects[row[0]]['idx'] = num_projects
                    num_projects += 1
                    for it, key in enumerate(header[1:]):
                        projects[row[0]][key.strip()] = row[it + 1].strip()
                elif section == 'votes':
                    votes[row[0]] = {}
                    for it, key in enumerate(header[1:]):
                        votes[row[0]][key.strip()] = row[it + 1].strip()
                    utility_matrix[num_voters] = np.zeros(num_projects)
                    for p in votes[row[0]]['vote'].split(','):
                        utility_matrix[num_voters][projects[p]['idx']] = np.random.uniform(0.85, 1.15, 1)
                    num_voters += 1

            if not os.path.exists(out_path):
                os.makedirs(out_path)

            with open(out_path + '/numbers.txt', 'w+') as num_file:
                num_file.write(str(num_voters) + '\n')
                num_file.write(str(num_projects) + '\n')
                num_file.write(meta['budget'] + '\n')

            with open(out_path + '/items_sizes.txt', 'w+') as items_file:
                for p in projects:
                    items_file.write(projects[p]['cost'] + ' ')

            with open(out_path + '/utility_matrix.txt', 'w') as util_file:
                for u in utility_matrix:
                    for item in utility_matrix[u]:
                        util_file.write(str(item) + ' ')
                    util_file.write('\n')


def read_saved_scenario(folder_name):
    with open(folder_name + '/numbers.txt', 'r') as num_file:
        num_agents = int(num_file.readline())
        num_items = int(num_file.readline())
        capacity = int(num_file.readline())
        # print(num_agents, num_items, capacity)

    with open(folder_name + '/items_sizes.txt', 'r') as items_file:
        line = items_file.readline()
        items_sizes = np.array([float(item) for item in line.split(' ')[:-1]]) / capacity
        # print(items_sizes)

    capacity = 1.0

    with open(folder_name + '/utility_matrix.txt', 'r') as util_file:
        utility_matrix = []

        for line in util_file:
            utility_vector = np.array([float(item) for item in line.split(' ')[:-1]])
            utility_matrix.append(utility_vector)

        utility_matrix = np.array(utility_matrix)

    assert (num_agents, num_items) == utility_matrix.shape
    assert num_items == len(items_sizes)
    return num_agents, num_items, capacity, items_sizes, utility_matrix


if __name__ == '__main__':
    process_raw_data(raw_data_folder='raw_data', final_data_folder='final_data')

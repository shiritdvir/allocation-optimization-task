import random
import pandas as pd
import matplotlib.pyplot as plt
import constants as c
import os

class World:
    organisms_dict = {}


class Allocation:

    number = 0
    surgeries_data_path = "C:/Documents/Greatmix.AI/surgeries.csv"
    surgeries_df = pd.read_csv(surgeries_data_path, index_col=0)
    surgeries_df['start'] = pd.to_datetime(surgeries_df['start'])
    surgeries_df['end'] = pd.to_datetime(surgeries_df['end'])

    def __init__(self, solution=None, n_rooms=c.N_ROOMS, n_anesthesiologist=c.N_ANESTHESIOLOGIST, dict_entry=True):
        if dict_entry:
            self.number = Allocation.number
            Allocation.number += 1

        self.n_surgeries = len(Allocation.surgeries_df)
        self.mutation_chance = c.MUTATION_CHANCE
        self.n_rooms = n_rooms
        self.n_anesthesiologist = n_anesthesiologist

        if solution is not None:
            self.solution = solution
        else:
            self.solution = self._random_solution()

        if dict_entry:
            World.organisms_dict[self.number] = self

    def _random_solution(self):
        random_solution = self.surgeries_df.copy()
        random_solution['anesthesiologist_id'] = random.choices(range(self.n_anesthesiologist), k=self.n_surgeries)
        random_solution['room_id'] = random.choices(range(self.n_rooms), k=self.n_surgeries)
        return random_solution

    def _crossover(self, partner1, partner2):
        child = pd.DataFrame()
        for i in range(len(partner1)):
            parent_chromosome = random.randint(0, 1)
            if parent_chromosome == 0:
                row = partner1.loc[i]
            elif parent_chromosome == 1:
                row = partner2.loc[i]
            child = pd.concat([child, row.to_frame().transpose()], ignore_index=True)
        return child

    def _crossover2(self, partner1, partner2):
        i = random.randint(0, len(partner1))
        child = pd.concat([partner1[:i], partner2[i:]], ignore_index=True)
        return child

    def _mutate(self, child):
        for i in range(len(child)):
            if random.random() < self.mutation_chance:
                child.loc[i, 'anesthesiologist_id'] = random.randint(0, self.n_anesthesiologist)
            if random.random() < self.mutation_chance:
                child.loc[i, 'room_id'] = random.randint(0, self.n_rooms)
        return

    def _mutate2(self, child):
        n_mutations = int(len(child) * self.mutation_chance)
        for i in range(n_mutations):
            child.loc[random.randint(0, len(child) - 1), 'anesthesiologist_id'] = \
                child.loc[random.randint(0, len(child)-1), 'anesthesiologist_id']
            child.loc[random.randint(0, len(child) - 1), 'room_id'] = \
                child.loc[random.randint(0, len(child)-1), 'room_id']
        return child

    def mate(self, partner, dict_entry=True):
        solution1 = self.solution
        solution2 = partner.solution
        child_solution = self._crossover2(solution1, solution2)
        child_solution = self._mutate2(child_solution)
        child_allocation = Allocation(solution=child_solution, dict_entry=dict_entry)
        return child_allocation

    def _anesthesiologist_validity(self, min_time=c.MIN_TIME, penalty=c.PENALTY):
        score = 0
        solution_sorted = self.solution.sort_values(by=['anesthesiologist_id', 'start'])
        for i in range(len(solution_sorted) - 1):
            current_row = solution_sorted.iloc[i]
            next_row = solution_sorted.iloc[i + 1]
            if current_row['anesthesiologist_id'] == next_row['anesthesiologist_id']:
                if current_row['end'] > next_row['start']:
                    score += penalty
                if current_row['room_id'] != next_row['room_id'] and current_row['end'] + pd.Timedelta(minutes=min_time) > next_row['start']:
                    score += penalty
        return score

    def _room_validity(self, penalty=c.PENALTY):
        score = 0
        solution_sorted = self.solution.sort_values(by=['room_id', 'start'])
        for i in range(len(solution_sorted) - 1):
            current_row = solution_sorted.iloc[i]
            next_row = solution_sorted.iloc[i + 1]
            if current_row['end'] > next_row['start'] and current_row['room_id'] == next_row['room_id']:
                score += penalty
        return score

    def _get_durations(self):
        solution_sorted = self.solution.sort_values(by=['anesthesiologist_id', 'start'])
        durations = []
        for id in set(solution_sorted['anesthesiologist_id']):
            id_df = solution_sorted[solution_sorted['anesthesiologist_id'] == id]
            duration = list(id_df['end'])[-1] - list(id_df['start'])[0]
            durations.append(duration.total_seconds() / 3600)
        return durations

    def _duration_score(self, durations, penalty=c.PENALTY):
        score = 0
        for duration in durations:
            if duration < 5 or duration > 12:
                score += penalty
            else:
                score += max(5, duration) + 0.5 * max(0, duration-9)
        return score

    def _get_fitness(self):
        score = 0
        score += self._anesthesiologist_validity()
        score += self._room_validity()
        durations = self._get_durations()
        return score + self._duration_score(durations)


class Population:

    def __init__(self, results_dir, n_population=100, n_generations=100, maximize=True):
        self.results_dir = results_dir
        self.n_population = n_population
        self.n_generations = n_generations
        self.maximize = maximize

        self.population = self.initialize_population()
        self.fitness_plot = []

    def initialize_population(self):
        population = []
        for i in range(self.n_population):
            solution = Allocation()
            population.append(solution)
        return population

    def selection(self, n_selected):
        selection = []
        fitness_list = [(allocation.number, allocation._get_fitness()) for allocation in self.population]
        sorted_values = sorted(fitness_list, key=lambda x: x[1], reverse=self.maximize)
        print(sorted_values)
        for n in range(n_selected):
            selection.append(World.organisms_dict[sorted_values[n][0]])
        return selection

    def next_generation(self):
        selection = self.selection(n_selected=c.N_SELECTION)
        self.population = []
        for pop_idx in range(self.n_population):
            i1 = random.randint(0, len(selection)-1)
            i2 = random.randint(0, len(selection)-1)
            self.population.append(selection[i1].mate(selection[i2]))
        fitness_list = [s._get_fitness() for s in self.population]
        self.fitness_plot.append(sum(fitness_list) / len(fitness_list))

    def plot_fitness(self):
        plt.plot(self.fitness_plot)
        plt.title('Average Fitness')
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.legend(['Fitness'], loc='lower right')
        plt.savefig(os.path.join(self.results_dir, "fitness_plot.png"))
        plt.clf()

    def run(self):
        for generation in range(self.n_generations):
            print(f"Generation {generation}")
            self.next_generation()


if __name__ == "__main__":

    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Run genetic algorithm to optimize schedule
    pop = Population(results_dir=results_dir, n_population=c.N_POPULATION, n_generations=c.N_GENERATIONS, maximize=c.MAXIMIZE)
    pop.run()
    pop.plot_fitness()

    # get best solution
    fitness_list = [(allocation.number, allocation._get_fitness()) for allocation in pop.population]
    sorted_values = sorted(fitness_list, key=lambda x: x[1], reverse=c.MAXIMIZE)
    best_solution = World.organisms_dict[sorted_values[0][0]]
    best_solution.solution.to_csv(os.path.join(results_dir, "best_solution.csv"))
    print('Best solution fitness: ', sorted_values[0][1])

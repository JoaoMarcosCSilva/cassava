from genome import *
import json
import copy

class Population:
    def __init__(self, size):
        self.individuals = [[Genome() for i in range(size)]]
        self.evaluations = []

    def generation(self, mutation_rate, train_func):
        evaluation = [g.evaluate(train_func) for g in self.individuals[-1]]
        self.evaluations.append(evaluation)
        
        best_individual = np.argmax(evaluation)

        new_individuals = copy.deepcopy(self.individuals[-1])

        for i in range(len(new_individuals)):
            if i == best_individual:
                continue
            else:
                new_individuals[i] = crossover(new_individuals[i], new_individuals[best_individual])
                new_individuals[i].mutate(mutation_rate)
        self.individuals.append(new_individuals)

    def save(self, filename):
        d = copy.deepcopy(self.__dict__)
        for i in range(len(d['individuals'])):
            for j in range(len(d['individuals'][i])):
                d['individuals'][i][j] = d['individuals'][i][j].__dict__

        with open(filename, 'w') as f:
            json.dump(d, f, indent = 2)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.__dict__ = json.load(f)
        for i in range(len(self.individuals)):
            for j in range(len(self.individuals[i])):
                k = Genome()
                k.__dict__ = self.individuals[i][j]
                self.individuals[i][j] = k
                


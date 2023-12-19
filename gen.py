import numpy as np
import random

# Словарь с приблизительными расстояниями между городами
distances = {
    'Москва': {'Париж': 2485, 'Токио': 7472, 'Нью-Йорк': 7537, 'Сидней': 14598, 'Рио-де-Жанейро': 10856},
    'Париж': {'Москва': 2485, 'Токио': 9711, 'Нью-Йорк': 5836, 'Сидней': 16983, 'Рио-де-Жанейро': 9187},
    'Токио': {'Москва': 7472, 'Париж': 9711, 'Нью-Йорк': 10855, 'Сидней': 7831, 'Рио-де-Жанейро': 18513},
    'Нью-Йорк': {'Москва': 7537, 'Париж': 5836, 'Токио': 10855, 'Сидней': 15980, 'Рио-де-Жанейро': 7711},
    'Сидней': {'Москва': 14598, 'Париж': 16983, 'Токио': 7831, 'Нью-Йорк': 15980, 'Рио-де-Жанейро': 13381},
    'Рио-де-Жанейро': {'Москва': 10856, 'Париж': 9187, 'Токио': 18513, 'Нью-Йорк': 7711, 'Сидней': 13381}
}


# Функция для расчета расстояния
def calculate_total_distance(route, distances_data):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distances_data[route[i]][route[i + 1]]
    total_distance += distances_data[route[-1]][route[0]]  # Замыкание маршрута
    return total_distance

# Генетический алгоритм

# Инициализация начальной популяции
def initialize_population(cities, population_size):
    population = []
    for _ in range(population_size):
        route = cities[:]
        random.shuffle(route)
        population.append(route)
    return population


# Оценка пригодности
def fitness(route, distances_data):
    return -calculate_total_distance(route, distances_data)


# Селекция
def selection(population, fitness_scores, num_parents):
    parents_indices = np.argsort(fitness_scores)[-num_parents:] # 50 кратчайших маршрутов indices
    return [population[i] for i in parents_indices] # 50 shortest actual routes


# Скрещивание
def crossover(parent1, parent2):
    childP1 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


# Мутация
def mutate(route, mutation_rate):
    for swapped in range(len(route)):
        if random.random() < mutation_rate:
            swapWith = int(random.random() * len(route))

            city1 = route[swapped]
            city2 = route[swapWith]

            route[swapped] = city2
            route[swapWith] = city1
    return route


# Создание нового поколения
def create_new_generation(population, fitness_scores, num_parents, mutation_rate):
    new_generation = []
    parents = selection(population, fitness_scores, num_parents)
    num_new_children = len(population) - num_parents

    for i in range(num_new_children):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = crossover(parent1, parent2)
        new_generation.append(mutate(child, mutation_rate))

    new_generation.extend(parents)
    return new_generation

cities = ['Москва', 'Париж', 'Токио', 'Нью-Йорк', 'Сидней', 'Рио-де-Жанейро']


DISTANCE_DATA = distances

# Параметры алгоритма
population_size = 100
num_parents = int(population_size / 2)
mutation_rate = 0.01

# Инициализация
population = initialize_population(cities, population_size)


# Генетический алгоритм
for generation in range(population_size):
    fitness_scores = [fitness(route, DISTANCE_DATA) for route in population]
    population = create_new_generation(population, fitness_scores, num_parents, mutation_rate)

# Результаты
best_route_index = np.argmax(fitness_scores)
best_route = population[best_route_index]
best_route_distance = -fitness_scores[best_route_index]

print("Лучший путь:", best_route)
print("Лучшая дистанция:", best_route_distance)

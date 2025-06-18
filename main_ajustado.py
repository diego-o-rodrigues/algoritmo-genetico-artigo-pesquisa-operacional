import pandas as pd
import numpy as np
import random
import itertools
import time
import multiprocessing
import os

# Carregar instâncias dos clusters
try:
    df1 = pd.read_excel("Matriz_Reconstruida_Euclidiana.xlsx", index_col=0)
    df2 = pd.read_excel("Cluster2_Distancias.xlsx", index_col=0)
except FileNotFoundError:
    print("Erro: Verifique se os arquivos 'Matriz_Reconstruida_Euclidiana.xlsx' e 'Cluster2_Distancias.xlsx' estão na mesma pasta do script.")
    exit()

# Função para calcular o custo de uma rota
def route_cost(route, dist_matrix):
    cost = 0
    for i in range(len(route) - 1):
        cost += dist_matrix.iloc[route[i], route[i + 1]]
    cost += dist_matrix.iloc[route[-1], route[0]]  # retorno ao depósito
    return cost

# NOVO OPERADOR: 2-opt para refinamento local (mais rápido que 3-opt)
def two_opt(route, dist_matrix):
    best = list(route) # Trabalha com uma lista mutável
    improved = True
    while improved:
        improved = False
        n = len(best)
        if n < 3: # 2-opt requer pelo menos 3 nós (depósito + 2 clientes)
            return best

        for i in range(1, n - 1): # Primeiro ponto de corte (não inclui o depósito)
            for j in range(i + 1, n): # Segundo ponto de corte (não inclui o depósito)
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                # Garante que o depósito (índice 0) não seja afetado por inversões
                # se i=0, o 0 seria invertido, o que não queremos.
                # Como i começa de 1, isso já é garantido.

                current_cost = route_cost(best, dist_matrix)
                new_cost = route_cost(new_route, dist_matrix)

                if new_cost < current_cost:
                    best = new_route
                    improved = True
        route = best
    return tuple(best) # Retorna como tupla para imutabilidade em multiprocessing

# Geração de população inicial
def initial_population(size, nodes):
    population = []
    for _ in range(size):
        perm = list(nodes) # nodes não inclui o depósito (0)
        random.shuffle(perm)
        population.append([0] + perm) # Depósito é sempre o primeiro
    return population

# Seleção por torneio
def tournament_selection(pop_scores, k=3):
    selected = random.sample(pop_scores, k)
    selected.sort(key=lambda x: x[1]) 
    return selected[0][0] 

# Crossover de ordem (OX)
def order_crossover(p1, p2):
    size = len(p1)
    if size < 3: # Rota muito pequena para crossover útil (depósito + 2 clientes)
        return random.choice([p1, p2]) 
    
    start, end = sorted(random.sample(range(1, size), 2)) 
    
    middle = p1[start:end] 
    
    p2_clients_ordered = [x for x in p2 if x != 0 and x not in middle]
    
    child = [0] * size 
    child[start:end] = middle
    
    current_p2_client_idx = 0
    for i in range(1, size): 
        if i < start or i >= end:
            child[i] = p2_clients_ordered[current_p2_client_idx]
            current_p2_client_idx += 1
            
    return child

# Mutação por troca simples
def swap_mutation(route):
    r = list(route) 
    if len(r) > 2: 
        i, j = random.sample(range(1, len(r)), 2)
        r[i], r[j] = r[j], r[i]
    return tuple(r) 

# Função para salvar os resultados em arquivo
def save_results(cluster_name, best_route, best_cost, execution_time):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"resultados_{cluster_name}_{timestamp}.txt"
    
    if not os.path.exists("resultados"):
        os.makedirs("resultados")
        
    filepath = os.path.join("resultados", filename)

    with open(filepath, "w") as f:
        f.write(f"Resultados para {cluster_name}:\n")
        f.write(f"Melhor Custo Encontrado: {best_cost:.2f}\n")
        f.write(f"Tempo de Execução Total: {execution_time:.2f} segundos\n")
        f.write(f"Melhor Rota Encontrada: {best_route}\n")
        f.write("\n")
    print(f"Resultados salvos em '{filepath}'")

# Algoritmo Genético com 2-opt (anteriormente 3-opt)
def genetic_algorithm(dist_matrix, generations=200, pop_size=50, mutation_rate=0.2, num_processors=None):
    nodes = list(range(1, len(dist_matrix))) 
    population = initial_population(pop_size, nodes)
    best_route = None
    best_cost = float("inf")

    if num_processors is None:
        num_processors = max(1, int(multiprocessing.cpu_count() * 0.8)) 

    with multiprocessing.Pool(processes=num_processors) as pool:
        start_time_total = time.time() 
        for generation in range(generations):
            generation_start_time = time.time() 

            # Calcula os scores em paralelo
            population_for_cost_calc = [(tuple(ind), dist_matrix) for ind in population]
            scores = pool.starmap(route_cost, population_for_cost_calc)
            
            pop_scores = list(zip(population, scores)) 

            new_population = []
            
            for _ in range(pop_size):
                parent1 = tournament_selection(pop_scores)
                parent2 = tournament_selection(pop_scores)
                
                child = order_crossover(parent1, parent2)

                if random.random() < mutation_rate:
                    child = swap_mutation(child)
                
                new_population.append(child)

            # Aplica 2-opt em paralelo para a nova população (agora two_opt)
            new_population_for_opt = [(tuple(ind), dist_matrix) for ind in new_population]
            refined_population = list(pool.starmap(two_opt, new_population_for_opt)) # Alterado para two_opt

            # Atualiza a melhor rota e custo
            for ind_refined in refined_population:
                cost = route_cost(ind_refined, dist_matrix)
                if cost < best_cost:
                    best_cost = cost
                    best_route = ind_refined
            
            population = refined_population 

            generation_end_time = time.time() 
            generation_time = generation_end_time - generation_start_time

            print(f"Geração {generation + 1:03d} / {generations:03d} | Melhor Custo = {best_cost:.2f} | Tempo da Geração: {generation_time:.2f}s")

        end_time_total = time.time() 
        execution_time_total = end_time_total - start_time_total

    return best_route, best_cost, execution_time_total


if __name__ == "__main__":
    num_cpus = multiprocessing.cpu_count()
    num_processors_to_use = max(1, int(num_cpus * 0.8))
    print(f"Usando {num_processors_to_use} de {num_cpus} núcleos lógicos ({num_processors_to_use/num_cpus:.0%}).")

    # Manter o número de gerações e população razoáveis
    # Para acelerar, podemos reduzir um pouco estes valores se necessário.
    GA_GENERATIONS = 500
    GA_POP_SIZE = 100    
    GA_MUTATION_RATE = 0.1 

    print("\nRodando AG para o Cluster 1...")
    best_route_1, best_cost_1, time_1 = genetic_algorithm(
        df1, 
        generations=GA_GENERATIONS, 
        pop_size=GA_POP_SIZE, 
        mutation_rate=GA_MUTATION_RATE, 
        num_processors=num_processors_to_use
    )
    print(f"\n--- Resultados Finais Cluster 1 ---")
    print(f"Melhor rota Cluster 1: {best_route_1} com custo {best_cost_1:.2f}")
    print(f"Tempo de execução total Cluster 1: {time_1:.2f} segundos\n")
    save_results("Cluster1", best_route_1, best_cost_1, time_1)

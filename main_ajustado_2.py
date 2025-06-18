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

# OTIMIZAÇÃO: Converter para array NumPy para acesso mais rápido na função de custo
# Isso evita a sobrecarga de iloc do Pandas em cada acesso.
dist_matrix_np1 = df1.values
dist_matrix_np2 = df2.values


# Função para calcular o custo de uma rota (AGORA USA NUMPY ARRAY)
def route_cost(route, dist_matrix_np): # Recebe a matriz NumPy
    cost = 0
    for i in range(len(route) - 1):
        cost += dist_matrix_np[route[i], route[i + 1]] # Acesso direto ao NumPy array
    cost += dist_matrix_np[route[-1], route[0]]  # retorno ao depósito
    return cost

# NOVO OPERADOR: 2-opt "First-Improvement" para refinamento local
# Retorna a rota assim que uma melhoria é encontrada.
def two_opt_first_improvement(route, dist_matrix_np): # Recebe a matriz NumPy
    best = list(route) 
    n = len(best)
    if n < 3: 
        return tuple(best) # Não pode otimizar rota muito pequena

    # Loop principal para continuar buscando melhorias enquanto elas forem encontradas
    while True:
        improved_this_iteration = False
        for i in range(1, n - 1): # Primeiro ponto de corte (não inclui o depósito)
            for j in range(i + 1, n): # Segundo ponto de corte (não inclui o depósito)
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                
                current_cost = route_cost(best, dist_matrix_np)
                new_cost = route_cost(new_route, dist_matrix_np)

                if new_cost < current_cost:
                    best = new_route
                    improved_this_iteration = True
                    # PARA AQUI E REINICIA A BUSCA: FIRST-IMPROVEMENT
                    break # Sai do loop 'j'
            if improved_this_iteration:
                break # Sai do loop 'i'
        
        if not improved_this_iteration:
            # Nenhuma melhoria foi encontrada em toda a iteração, então estamos em um ótimo local (2-opt)
            break
    
    return tuple(best) 

# Geração de população inicial
def initial_population(size, nodes):
    population = []
    for _ in range(size):
        perm = list(nodes) 
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
    if size < 3: 
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

    # AQUI: Adicionar encoding='utf-8' ao abrir o arquivo
    with open(filepath, "w", encoding='utf-8') as f:
        f.write(f"Resultados para {cluster_name}:\n")
        f.write(f"Melhor Custo Encontrado: {best_cost:.2f}\n")
        f.write(f"Tempo de Execução Total: {execution_time:.2f} segundos\n")
        f.write(f"Melhor Rota Encontrada: {best_route}\n")
        f.write("\n")
    print(f"Resultados salvos em '{filepath}'")
    
# Algoritmo Genético principal
def genetic_algorithm(dist_matrix_np, generations=200, pop_size=50, mutation_rate=0.2, num_processors=None):
    # 'nodes' são apenas os clientes, sem o depósito
    nodes = list(range(1, dist_matrix_np.shape[0])) # Usa shape[0] para obter o número de nós
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
            # Importante: dist_matrix_np deve ser passado para cada processo.
            # Se for muito grande, considere usar um Manager para Shared Memory, mas para matrizes de distância típicas, passar por argumento funciona.
            population_for_cost_calc = [(tuple(ind), dist_matrix_np) for ind in population]
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

            # Aplica 2-opt first-improvement em paralelo para a nova população
            new_population_for_opt = [(tuple(ind), dist_matrix_np) for ind in new_population]
            # CHAMA A VERSÃO FIRST-IMPROVEMENT
            refined_population = list(pool.starmap(two_opt_first_improvement, new_population_for_opt)) 

            # Atualiza a melhor rota e custo
            for ind_refined in refined_population:
                cost = route_cost(ind_refined, dist_matrix_np) # Calcula custo com a matriz NumPy
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

    # PARÂMETROS MAIS AGRESSIVOS PARA VELOCIDADE
    GA_GENERATIONS = 150 # Reduzido de 500
    GA_POP_SIZE = 50     # Reduzido de 100
    GA_MUTATION_RATE = 0.1 # Mantido, mas pode ser ajustado se o AG estiver estagnando muito rápido

    print("\nAlgoritmo Genético Iniciado...\n")
    # Passa a matriz NumPy diretamente
    best_route_1, best_cost_1, time_1 = genetic_algorithm(
        dist_matrix_np1, # Passa o array NumPy
        generations=GA_GENERATIONS, 
        pop_size=GA_POP_SIZE, 
        mutation_rate=GA_MUTATION_RATE, 
        num_processors=num_processors_to_use
    )
    print(f"\n--- Resultados Finais do Algoritmo Genético---")
    print(f"Melhor rota proposta: {best_route_1} com custo {best_cost_1:.2f}")
    print(f"Tempo de execução total: {time_1:.2f} segundos\n")
    save_results("rotas", best_route_1, best_cost_1, time_1)


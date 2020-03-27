
class Criteria:
    RESPONSE_TIME = 1
class NetworkPrioritization:
    #No se necesita que sea estatico
    #@staticmethod
    
    #Uso de self, podría ser lo adecuado
    def prioritize_network(self, response_times: list, cost: list, priority: Criteria) -> list:
        if priority == Criteria.RESPONSE_TIME:
            first_priority = enumerate(response_times)
            second_priority = enumerate(cost)
        else:
            first_priority = enumerate(cost)
            second_priority = enumerate(response_times)
        #Enumera y organiza la lista que tiene la prioridad
        sorted_list = sorted(first_priority, key=lambda x: x[1])
        print(f"sorted list: {sorted_list}")
        
        #Prioridad, los indices
        keys = list()
        for key, network in sorted_list:
            if key not in keys:
                print("Se valida si hay un valor de red repetido")
                print(response_times.count(network))
                if response_times.count(network) > 1:
                    print("Almacena los duplicados")
                    duplicated = [first_network_tuple[0] for first_network_tuple in sorted_list if first_network_tuple[1] == network]
                    print(duplicated)
                    print("Obtiene los valores de los indices repetidos segun la 'segunda' prioridad")
                    sub_list = [second_network_tuple for second_network_tuple in second_priority if second_network_tuple[0] in duplicated]
                    print(sub_list)
                    print("Obtiene los indices ordenados usando la segunda prioridad como apoyo")
                    sort_sub_list = [second_network_tuple[0] for second_network_tuple in sorted(sub_list, key=lambda x: x[1])]
                    print(sort_sub_list)
                    print("Agrega las repetidas en el orden correspondiente")
                    keys = keys + sort_sub_list
                    print(f"Estado actual del proceso de priorización {keys}")
                else:
                    keys.append(key)
        return keys
network = NetworkPrioritization()
network.prioritize_network([20, 15, 100, 15, 50, 9], [50, 60, 30, 55, 40, 70], Criteria.RESPONSE_TIME)


def fed_avg(client_params_list, client_sizes):
    aggregated_params = {}
    total_size = sum(client_sizes)
    for k in client_params_list[0].keys():
        aggregated_params[k] = sum(client_params[k].float() * client_sizes[i] / total_size 
                                 for i, client_params in enumerate(client_params_list))
    return aggregated_params

static void phys_map_node_reserve(unsigned nodes)

{

    if (next_map.nodes_nb + nodes > next_map.nodes_nb_alloc) {

        next_map.nodes_nb_alloc = MAX(next_map.nodes_nb_alloc * 2,

                                            16);

        next_map.nodes_nb_alloc = MAX(next_map.nodes_nb_alloc,

                                      next_map.nodes_nb + nodes);

        next_map.nodes = g_renew(Node, next_map.nodes,

                                 next_map.nodes_nb_alloc);

    }

}

static uint32_t phys_map_node_alloc(void)

{

    unsigned i;

    uint32_t ret;



    ret = next_map.nodes_nb++;

    assert(ret != PHYS_MAP_NODE_NIL);

    assert(ret != next_map.nodes_nb_alloc);

    for (i = 0; i < P_L2_SIZE; ++i) {

        next_map.nodes[ret][i].skip = 1;

        next_map.nodes[ret][i].ptr = PHYS_MAP_NODE_NIL;

    }

    return ret;

}

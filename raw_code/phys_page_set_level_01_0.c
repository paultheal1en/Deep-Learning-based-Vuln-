static void phys_page_set_level(PhysPageEntry *lp, target_phys_addr_t *index,

                                target_phys_addr_t *nb, uint16_t leaf,

                                int level)

{

    PhysPageEntry *p;

    int i;

    target_phys_addr_t step = (target_phys_addr_t)1 << (level * L2_BITS);



    if (!lp->is_leaf && lp->ptr == PHYS_MAP_NODE_NIL) {

        lp->ptr = phys_map_node_alloc();

        p = phys_map_nodes[lp->ptr];

        if (level == 0) {

            for (i = 0; i < L2_SIZE; i++) {

                p[i].is_leaf = 1;

                p[i].ptr = phys_section_unassigned;

            }

        }

    } else {

        p = phys_map_nodes[lp->ptr];

    }

    lp = &p[(*index >> (level * L2_BITS)) & (L2_SIZE - 1)];



    while (*nb && lp < &p[L2_SIZE]) {

        if ((*index & (step - 1)) == 0 && *nb >= step) {

            lp->is_leaf = true;

            lp->ptr = leaf;

            *index += step;

            *nb -= step;

        } else {

            phys_page_set_level(lp, index, nb, leaf, level - 1);

        }

        ++lp;

    }

}

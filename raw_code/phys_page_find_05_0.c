MemoryRegionSection *phys_page_find(target_phys_addr_t index)

{

    PhysPageEntry lp = phys_map;

    PhysPageEntry *p;

    int i;

    uint16_t s_index = phys_section_unassigned;



    for (i = P_L2_LEVELS - 1; i >= 0 && !lp.is_leaf; i--) {

        if (lp.ptr == PHYS_MAP_NODE_NIL) {

            goto not_found;

        }

        p = phys_map_nodes[lp.ptr];

        lp = p[(index >> (i * L2_BITS)) & (L2_SIZE - 1)];

    }



    s_index = lp.ptr;

not_found:

    return &phys_sections[s_index];

}

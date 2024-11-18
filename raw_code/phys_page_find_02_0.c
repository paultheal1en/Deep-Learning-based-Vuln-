MemoryRegionSection *phys_page_find(AddressSpaceDispatch *d, hwaddr index)

{

    PhysPageEntry lp = d->phys_map;

    PhysPageEntry *p;

    int i;



    for (i = P_L2_LEVELS - 1; i >= 0 && !lp.is_leaf; i--) {

        if (lp.ptr == PHYS_MAP_NODE_NIL) {

            return &phys_sections[phys_section_unassigned];

        }

        p = phys_map_nodes[lp.ptr];

        lp = p[(index >> (i * L2_BITS)) & (L2_SIZE - 1)];

    }

    return &phys_sections[lp.ptr];

}

static MemoryRegionSection *phys_page_find(PhysPageEntry lp, hwaddr addr,

                                           Node *nodes, MemoryRegionSection *sections)

{

    PhysPageEntry *p;

    hwaddr index = addr >> TARGET_PAGE_BITS;

    int i;



    for (i = P_L2_LEVELS; lp.skip && (i -= lp.skip) >= 0;) {

        if (lp.ptr == PHYS_MAP_NODE_NIL) {

            return &sections[PHYS_SECTION_UNASSIGNED];

        }

        p = nodes[lp.ptr];

        lp = p[(index >> (i * P_L2_BITS)) & (P_L2_SIZE - 1)];

    }

    return &sections[lp.ptr];

}

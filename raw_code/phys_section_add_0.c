static uint16_t phys_section_add(MemoryRegionSection *section)

{

    /* The physical section number is ORed with a page-aligned

     * pointer to produce the iotlb entries.  Thus it should

     * never overflow into the page-aligned value.

     */

    assert(next_map.sections_nb < TARGET_PAGE_SIZE);



    if (next_map.sections_nb == next_map.sections_nb_alloc) {

        next_map.sections_nb_alloc = MAX(next_map.sections_nb_alloc * 2,

                                         16);

        next_map.sections = g_renew(MemoryRegionSection, next_map.sections,

                                    next_map.sections_nb_alloc);

    }

    next_map.sections[next_map.sections_nb] = *section;

    memory_region_ref(section->mr);

    return next_map.sections_nb++;

}

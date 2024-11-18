static void phys_sections_clear(PhysPageMap *map)

{

    while (map->sections_nb > 0) {

        MemoryRegionSection *section = &map->sections[--map->sections_nb];

        phys_section_destroy(section->mr);

    }

    g_free(map->sections);

    g_free(map->nodes);

}

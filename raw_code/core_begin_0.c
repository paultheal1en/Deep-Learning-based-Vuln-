static void core_begin(MemoryListener *listener)

{

    destroy_all_mappings();

    phys_sections_clear();

    phys_map.ptr = PHYS_MAP_NODE_NIL;

    phys_section_unassigned = dummy_section(&io_mem_unassigned);

    phys_section_notdirty = dummy_section(&io_mem_notdirty);

    phys_section_rom = dummy_section(&io_mem_rom);

    phys_section_watch = dummy_section(&io_mem_watch);

}

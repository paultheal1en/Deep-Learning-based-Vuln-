void tb_invalidate_phys_addr(hwaddr addr)

{

    ram_addr_t ram_addr;

    MemoryRegionSection *section;



    section = phys_page_find(address_space_memory.dispatch,

                             addr >> TARGET_PAGE_BITS);

    if (!(memory_region_is_ram(section->mr)

          || memory_region_is_romd(section->mr))) {

        return;

    }

    ram_addr = (memory_region_get_ram_addr(section->mr) & TARGET_PAGE_MASK)

        + memory_region_section_addr(section, addr);

    tb_invalidate_phys_page_range(ram_addr, ram_addr + 1, 0);

}

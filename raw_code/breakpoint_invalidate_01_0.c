static void breakpoint_invalidate(CPUArchState *env, target_ulong pc)

{

    target_phys_addr_t addr;

    ram_addr_t ram_addr;

    MemoryRegionSection *section;



    addr = cpu_get_phys_page_debug(env, pc);

    section = phys_page_find(addr >> TARGET_PAGE_BITS);

    if (!(memory_region_is_ram(section->mr)

          || (section->mr->rom_device && section->mr->readable))) {

        return;

    }

    ram_addr = (memory_region_get_ram_addr(section->mr) & TARGET_PAGE_MASK)

        + section_addr(section, addr);

    tb_invalidate_phys_page_range(ram_addr, ram_addr + 1, 0);

}

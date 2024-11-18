bool cpu_physical_memory_is_io(target_phys_addr_t phys_addr)

{

    MemoryRegionSection *section;



    section = phys_page_find(phys_addr >> TARGET_PAGE_BITS);



    return !(memory_region_is_ram(section->mr) ||

             memory_region_is_romd(section->mr));

}

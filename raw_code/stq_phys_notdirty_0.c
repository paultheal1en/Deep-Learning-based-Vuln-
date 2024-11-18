void stq_phys_notdirty(target_phys_addr_t addr, uint64_t val)

{

    uint8_t *ptr;

    MemoryRegionSection *section;



    section = phys_page_find(addr >> TARGET_PAGE_BITS);



    if (!memory_region_is_ram(section->mr) || section->readonly) {

        addr = memory_region_section_addr(section, addr);

        if (memory_region_is_ram(section->mr)) {

            section = &phys_sections[phys_section_rom];

        }

#ifdef TARGET_WORDS_BIGENDIAN

        io_mem_write(section->mr, addr, val >> 32, 4);

        io_mem_write(section->mr, addr + 4, (uint32_t)val, 4);

#else

        io_mem_write(section->mr, addr, (uint32_t)val, 4);

        io_mem_write(section->mr, addr + 4, val >> 32, 4);

#endif

    } else {

        ptr = qemu_get_ram_ptr((memory_region_get_ram_addr(section->mr)

                                & TARGET_PAGE_MASK)

                               + memory_region_section_addr(section, addr));

        stq_p(ptr, val);

    }

}

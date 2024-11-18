static void subpage_write(void *opaque, target_phys_addr_t addr,

                          uint64_t value, unsigned len)

{

    subpage_t *mmio = opaque;

    unsigned int idx = SUBPAGE_IDX(addr);

    MemoryRegionSection *section;

#if defined(DEBUG_SUBPAGE)

    printf("%s: subpage %p len %d addr " TARGET_FMT_plx

           " idx %d value %"PRIx64"\n",

           __func__, mmio, len, addr, idx, value);

#endif



    section = &phys_sections[mmio->sub_section[idx]];

    addr += mmio->base;

    addr -= section->offset_within_address_space;

    addr += section->offset_within_region;

    io_mem_write(section->mr, addr, value, len);

}

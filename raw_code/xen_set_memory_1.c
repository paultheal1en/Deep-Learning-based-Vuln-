static void xen_set_memory(struct MemoryListener *listener,

                           MemoryRegionSection *section,

                           bool add)

{

    XenIOState *state = container_of(listener, XenIOState, memory_listener);

    hwaddr start_addr = section->offset_within_address_space;

    ram_addr_t size = int128_get64(section->size);

    bool log_dirty = memory_region_is_logging(section->mr);

    hvmmem_type_t mem_type;



    if (!memory_region_is_ram(section->mr)) {

        return;

    }



    if (!(section->mr != &ram_memory

          && ( (log_dirty && add) || (!log_dirty && !add)))) {

        return;

    }



    trace_xen_client_set_memory(start_addr, size, log_dirty);



    start_addr &= TARGET_PAGE_MASK;

    size = TARGET_PAGE_ALIGN(size);



    if (add) {

        if (!memory_region_is_rom(section->mr)) {

            xen_add_to_physmap(state, start_addr, size,

                               section->mr, section->offset_within_region);

        } else {

            mem_type = HVMMEM_ram_ro;

            if (xc_hvm_set_mem_type(xen_xc, xen_domid, mem_type,

                                    start_addr >> TARGET_PAGE_BITS,

                                    size >> TARGET_PAGE_BITS)) {

                DPRINTF("xc_hvm_set_mem_type error, addr: "TARGET_FMT_plx"\n",

                        start_addr);

            }

        }

    } else {

        if (xen_remove_from_physmap(state, start_addr, size) < 0) {

            DPRINTF("physmapping does not exist at "TARGET_FMT_plx"\n", start_addr);

        }

    }

}

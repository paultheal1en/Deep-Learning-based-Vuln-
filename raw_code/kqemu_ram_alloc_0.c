static ram_addr_t kqemu_ram_alloc(ram_addr_t size)

{

    ram_addr_t addr;

    if ((last_ram_offset + size) > kqemu_phys_ram_size) {

        fprintf(stderr, "Not enough memory (requested_size = %" PRIu64 ", max memory = %" PRIu64 ")\n",

                (uint64_t)size, (uint64_t)kqemu_phys_ram_size);

        abort();

    }

    addr = last_ram_offset;

    last_ram_offset = TARGET_PAGE_ALIGN(last_ram_offset + size);

    return addr;

}

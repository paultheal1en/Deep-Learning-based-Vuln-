void *cpu_physical_memory_map(target_phys_addr_t addr,

                              target_phys_addr_t *plen,

                              int is_write)

{

    return address_space_map(&address_space_memory, addr, plen, is_write);

}

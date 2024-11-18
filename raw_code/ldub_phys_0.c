uint32_t ldub_phys(target_phys_addr_t addr)

{

    uint8_t val;

    cpu_physical_memory_read(addr, &val, 1);

    return val;

}

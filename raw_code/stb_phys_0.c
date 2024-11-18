void stb_phys(target_phys_addr_t addr, uint32_t val)

{

    uint8_t v = val;

    cpu_physical_memory_write(addr, &v, 1);

}

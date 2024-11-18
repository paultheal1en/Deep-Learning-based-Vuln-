void stw_phys(target_phys_addr_t addr, uint32_t val)

{

    uint16_t v = tswap16(val);

    cpu_physical_memory_write(addr, (const uint8_t *)&v, 2);

}

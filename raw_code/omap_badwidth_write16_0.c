void omap_badwidth_write16(void *opaque, target_phys_addr_t addr,

                uint32_t value)

{

    uint16_t val16 = value;



    OMAP_16B_REG(addr);

    cpu_physical_memory_write(addr, (void *) &val16, 2);

}

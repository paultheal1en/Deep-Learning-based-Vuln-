static void sysbus_esp_mem_write(void *opaque, target_phys_addr_t addr,

                                 uint64_t val, unsigned int size)

{

    SysBusESPState *sysbus = opaque;

    uint32_t saddr;



    saddr = addr >> sysbus->it_shift;

    esp_reg_write(&sysbus->esp, saddr, val);

}

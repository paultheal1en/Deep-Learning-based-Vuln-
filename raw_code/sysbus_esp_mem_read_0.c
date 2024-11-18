static uint64_t sysbus_esp_mem_read(void *opaque, target_phys_addr_t addr,

                                    unsigned int size)

{

    SysBusESPState *sysbus = opaque;

    uint32_t saddr;



    saddr = addr >> sysbus->it_shift;

    return esp_reg_read(&sysbus->esp, saddr);

}

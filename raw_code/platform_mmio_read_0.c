static uint32_t platform_mmio_read(ReadWriteHandler *handler, pcibus_t addr, int len)

{

    DPRINTF("Warning: attempted read from physical address "

            "0x" TARGET_FMT_plx " in xen platform mmio space\n", addr);



    return 0;

}

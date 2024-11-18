static void platform_mmio_map(PCIDevice *d, int region_num,

                              pcibus_t addr, pcibus_t size, int type)

{

    int mmio_io_addr;



    mmio_io_addr = cpu_register_io_memory_simple(&platform_mmio_handler,

                                                 DEVICE_NATIVE_ENDIAN);



    cpu_register_physical_memory(addr, size, mmio_io_addr);

}

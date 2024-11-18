static int pci_unin_internal_init_device(SysBusDevice *dev)

{

    UNINState *s;

    int pci_mem_config, pci_mem_data;



    /* Uninorth internal bus */

    s = FROM_SYSBUS(UNINState, dev);



    pci_mem_config = cpu_register_io_memory(pci_unin_config_read,

                                            pci_unin_config_write, s);

    pci_mem_data = cpu_register_io_memory(pci_unin_read,

                                          pci_unin_write, s);

    sysbus_init_mmio(dev, 0x1000, pci_mem_config);

    sysbus_init_mmio(dev, 0x1000, pci_mem_data);

    return 0;

}

static int pci_unin_main_init_device(SysBusDevice *dev)

{

    UNINState *s;

    int pci_mem_config, pci_mem_data;



    /* Use values found on a real PowerMac */

    /* Uninorth main bus */

    s = FROM_SYSBUS(UNINState, dev);



    pci_mem_config = cpu_register_io_memory(pci_unin_main_config_read,

                                            pci_unin_main_config_write, s);

    pci_mem_data = cpu_register_io_memory(pci_unin_main_read,

                                          pci_unin_main_write, &s->host_state);



    sysbus_init_mmio(dev, 0x1000, pci_mem_config);

    sysbus_init_mmio(dev, 0x1000, pci_mem_data);



    register_savevm("uninorth", 0, 1, pci_unin_save, pci_unin_load, &s->host_state);

    qemu_register_reset(pci_unin_reset, &s->host_state);

    return 0;

}

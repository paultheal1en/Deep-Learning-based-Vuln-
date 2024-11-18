static int pci_grackle_init_device(SysBusDevice *dev)

{

    GrackleState *s;

    int pci_mem_config, pci_mem_data;



    s = FROM_SYSBUS(GrackleState, dev);



    pci_mem_config = cpu_register_io_memory(pci_grackle_config_read,

                                            pci_grackle_config_write, s);

    pci_mem_data = cpu_register_io_memory(pci_grackle_read,

                                          pci_grackle_write,

                                          &s->host_state);

    sysbus_init_mmio(dev, 0x1000, pci_mem_config);

    sysbus_init_mmio(dev, 0x1000, pci_mem_data);



    register_savevm("grackle", 0, 1, pci_grackle_save, pci_grackle_load,

                    &s->host_state);

    qemu_register_reset(pci_grackle_reset, &s->host_state);

    return 0;

}

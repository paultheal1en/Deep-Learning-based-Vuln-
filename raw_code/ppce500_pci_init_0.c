PCIBus *ppce500_pci_init(qemu_irq pci_irqs[4], target_phys_addr_t registers)

{

    PPCE500PCIState *controller;

    PCIDevice *d;

    int index;

    static int ppce500_pci_id;



    controller = qemu_mallocz(sizeof(PPCE500PCIState));



    controller->pci_state.bus = pci_register_bus(NULL, "pci",

                                                 mpc85xx_pci_set_irq,

                                                 mpc85xx_pci_map_irq,

                                                 pci_irqs, 0x88, 4);

    d = pci_register_device(controller->pci_state.bus,

                            "host bridge", sizeof(PCIDevice),

                            0, NULL, NULL);



    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_FREESCALE);

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_MPC8533E);

    pci_config_set_class(d->config, PCI_CLASS_PROCESSOR_POWERPC);



    controller->pci_dev = d;



    /* CFGADDR */

    index = cpu_register_io_memory(pcie500_cfgaddr_read,

                                   pcie500_cfgaddr_write, controller);

    if (index < 0)

        goto free;

    cpu_register_physical_memory(registers + PCIE500_CFGADDR, 4, index);



    /* CFGDATA */

    index = cpu_register_io_memory(pcie500_cfgdata_read,

                                   pcie500_cfgdata_write,

                                   &controller->pci_state);

    if (index < 0)

        goto free;

    cpu_register_physical_memory(registers + PCIE500_CFGDATA, 4, index);



    index = cpu_register_io_memory(e500_pci_reg_read,

                                   e500_pci_reg_write, controller);

    if (index < 0)

        goto free;

    cpu_register_physical_memory(registers + PCIE500_REG_BASE,

                                   PCIE500_REG_SIZE, index);



    /* XXX load/save code not tested. */

    register_savevm("ppce500_pci", ppce500_pci_id++, 1,

                    ppce500_pci_save, ppce500_pci_load, controller);



    return controller->pci_state.bus;



free:

    printf("%s error\n", __func__);

    qemu_free(controller);

    return NULL;

}

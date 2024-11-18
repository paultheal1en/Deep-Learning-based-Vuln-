PCIBus *ppc4xx_pci_init(CPUState *env, qemu_irq pci_irqs[4],

                        target_phys_addr_t config_space,

                        target_phys_addr_t int_ack,

                        target_phys_addr_t special_cycle,

                        target_phys_addr_t registers)

{

    PPC4xxPCIState *controller;

    int index;

    static int ppc4xx_pci_id;

    uint8_t *pci_conf;



    controller = qemu_mallocz(sizeof(PPC4xxPCIState));



    controller->pci_state.bus = pci_register_bus(NULL, "pci",

                                                 ppc4xx_pci_set_irq,

                                                 ppc4xx_pci_map_irq,

                                                 pci_irqs, 0, 4);



    controller->pci_dev = pci_register_device(controller->pci_state.bus,

                                              "host bridge", sizeof(PCIDevice),

                                              0, NULL, NULL);

    pci_conf = controller->pci_dev->config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_IBM);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_IBM_440GX);

    pci_config_set_class(pci_conf, PCI_CLASS_BRIDGE_OTHER);



    /* CFGADDR */

    index = cpu_register_io_memory(pci4xx_cfgaddr_read,

                                   pci4xx_cfgaddr_write, controller);

    if (index < 0)

        goto free;

    cpu_register_physical_memory(config_space + PCIC0_CFGADDR, 4, index);



    /* CFGDATA */

    index = cpu_register_io_memory(pci4xx_cfgdata_read,

                                   pci4xx_cfgdata_write,

                                   &controller->pci_state);

    if (index < 0)

        goto free;

    cpu_register_physical_memory(config_space + PCIC0_CFGDATA, 4, index);



    /* Internal registers */

    index = cpu_register_io_memory(pci_reg_read, pci_reg_write, controller);

    if (index < 0)

        goto free;

    cpu_register_physical_memory(registers, PCI_REG_SIZE, index);



    qemu_register_reset(ppc4xx_pci_reset, controller);



    /* XXX load/save code not tested. */

    register_savevm("ppc4xx_pci", ppc4xx_pci_id++, 1,

                    ppc4xx_pci_save, ppc4xx_pci_load, controller);



    return controller->pci_state.bus;



free:

    printf("%s error\n", __func__);

    qemu_free(controller);

    return NULL;

}

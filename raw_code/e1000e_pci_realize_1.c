static void e1000e_pci_realize(PCIDevice *pci_dev, Error **errp)

{

    static const uint16_t e1000e_pmrb_offset = 0x0C8;

    static const uint16_t e1000e_pcie_offset = 0x0E0;

    static const uint16_t e1000e_aer_offset =  0x100;

    static const uint16_t e1000e_dsn_offset =  0x140;

    E1000EState *s = E1000E(pci_dev);

    uint8_t *macaddr;

    int ret;



    trace_e1000e_cb_pci_realize();



    pci_dev->config_write = e1000e_write_config;



    pci_dev->config[PCI_CACHE_LINE_SIZE] = 0x10;

    pci_dev->config[PCI_INTERRUPT_PIN] = 1;



    pci_set_word(pci_dev->config + PCI_SUBSYSTEM_VENDOR_ID, s->subsys_ven);

    pci_set_word(pci_dev->config + PCI_SUBSYSTEM_ID, s->subsys);



    s->subsys_ven_used = s->subsys_ven;

    s->subsys_used = s->subsys;



    /* Define IO/MMIO regions */

    memory_region_init_io(&s->mmio, OBJECT(s), &mmio_ops, s,

                          "e1000e-mmio", E1000E_MMIO_SIZE);

    pci_register_bar(pci_dev, E1000E_MMIO_IDX,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);



    /*

     * We provide a dummy implementation for the flash BAR

     * for drivers that may theoretically probe for its presence.

     */

    memory_region_init(&s->flash, OBJECT(s),

                       "e1000e-flash", E1000E_FLASH_SIZE);

    pci_register_bar(pci_dev, E1000E_FLASH_IDX,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &s->flash);



    memory_region_init_io(&s->io, OBJECT(s), &io_ops, s,

                          "e1000e-io", E1000E_IO_SIZE);

    pci_register_bar(pci_dev, E1000E_IO_IDX,

                     PCI_BASE_ADDRESS_SPACE_IO, &s->io);



    memory_region_init(&s->msix, OBJECT(s), "e1000e-msix",

                       E1000E_MSIX_SIZE);

    pci_register_bar(pci_dev, E1000E_MSIX_IDX,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &s->msix);



    /* Create networking backend */

    qemu_macaddr_default_if_unset(&s->conf.macaddr);

    macaddr = s->conf.macaddr.a;



    e1000e_init_msix(s);



    if (pcie_endpoint_cap_v1_init(pci_dev, e1000e_pcie_offset) < 0) {

        hw_error("Failed to initialize PCIe capability");

    }



    ret = msi_init(PCI_DEVICE(s), 0xD0, 1, true, false, NULL);

    if (ret) {

        trace_e1000e_msi_init_fail(ret);

    }



    if (e1000e_add_pm_capability(pci_dev, e1000e_pmrb_offset,

                                  PCI_PM_CAP_DSI) < 0) {

        hw_error("Failed to initialize PM capability");

    }



    if (pcie_aer_init(pci_dev, e1000e_aer_offset, PCI_ERR_SIZEOF) < 0) {

        hw_error("Failed to initialize AER capability");

    }



    pcie_dev_ser_num_init(pci_dev, e1000e_dsn_offset,

                          e1000e_gen_dsn(macaddr));



    e1000e_init_net_peer(s, pci_dev, macaddr);



    /* Initialize core */

    e1000e_core_realize(s);



    e1000e_core_pci_realize(&s->core,

                            e1000e_eeprom_template,

                            sizeof(e1000e_eeprom_template),

                            macaddr);

}

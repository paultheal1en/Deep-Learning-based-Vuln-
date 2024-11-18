void pci_bridge_reset_reg(PCIDevice *dev)

{

    uint8_t *conf = dev->config;



    conf[PCI_PRIMARY_BUS] = 0;

    conf[PCI_SECONDARY_BUS] = 0;

    conf[PCI_SUBORDINATE_BUS] = 0;

    conf[PCI_SEC_LATENCY_TIMER] = 0;



    conf[PCI_IO_BASE] = 0;

    conf[PCI_IO_LIMIT] = 0;

    pci_set_word(conf + PCI_MEMORY_BASE, 0);

    pci_set_word(conf + PCI_MEMORY_LIMIT, 0);

    pci_set_word(conf + PCI_PREF_MEMORY_BASE, 0);

    pci_set_word(conf + PCI_PREF_MEMORY_LIMIT, 0);

    pci_set_word(conf + PCI_PREF_BASE_UPPER32, 0);

    pci_set_word(conf + PCI_PREF_LIMIT_UPPER32, 0);



    pci_set_word(conf + PCI_BRIDGE_CONTROL, 0);

}

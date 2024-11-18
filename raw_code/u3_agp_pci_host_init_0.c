static int u3_agp_pci_host_init(PCIDevice *d)

{

    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_APPLE);

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_APPLE_U3_AGP);

    /* revision */

    d->config[0x08] = 0x00;

    pci_config_set_class(d->config, PCI_CLASS_BRIDGE_HOST);

    /* cache line size */

    d->config[0x0C] = 0x08;

    /* latency timer */

    d->config[0x0D] = 0x10;

    return 0;

}

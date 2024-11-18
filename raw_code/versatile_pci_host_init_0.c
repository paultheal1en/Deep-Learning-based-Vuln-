static int versatile_pci_host_init(PCIDevice *d)

{

    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_XILINX);

    /* Both boards have the same device ID.  Oh well.  */

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_XILINX_XC2VP30);

    pci_set_word(d->config + PCI_STATUS,

		 PCI_STATUS_66MHZ | PCI_STATUS_DEVSEL_MEDIUM);

    pci_config_set_class(d->config, PCI_CLASS_PROCESSOR_CO);

    pci_set_byte(d->config + PCI_LATENCY_TIMER, 0x10);

    return 0;

}

static int xen_platform_initfn(PCIDevice *dev)
{
    PCIXenPlatformState *d = XEN_PLATFORM(dev);
    uint8_t *pci_conf;
    pci_conf = dev->config;
    pci_set_word(pci_conf + PCI_COMMAND, PCI_COMMAND_IO | PCI_COMMAND_MEMORY);
    pci_config_set_prog_interface(pci_conf, 0);
    pci_conf[PCI_INTERRUPT_PIN] = 1;
    platform_ioport_bar_setup(d);
    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_SPACE_IO, &d->bar);
    /* reserve 16MB mmio address for share memory*/
    platform_mmio_setup(d);
    pci_register_bar(dev, 1, PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &d->mmio_bar);
    platform_fixed_ioport_init(d);
    return 0;
}
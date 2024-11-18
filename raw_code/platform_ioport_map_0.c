static void platform_ioport_map(PCIDevice *pci_dev, int region_num, pcibus_t addr, pcibus_t size, int type)

{

    PCIXenPlatformState *d = DO_UPCAST(PCIXenPlatformState, pci_dev, pci_dev);



    register_ioport_write(addr, size, 1, xen_platform_ioport_writeb, d);

    register_ioport_read(addr, size, 1, xen_platform_ioport_readb, d);

}

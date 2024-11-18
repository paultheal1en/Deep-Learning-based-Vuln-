static int i440fx_pcihost_initfn(SysBusDevice *dev)

{

    I440FXState *s = FROM_SYSBUS(I440FXState, dev);



    register_ioport_write(0xcf8, 4, 4, i440fx_addr_writel, s);

    register_ioport_read(0xcf8, 4, 4, i440fx_addr_readl, s);



    register_ioport_write(0xcfc, 4, 1, pci_host_data_writeb, s);

    register_ioport_write(0xcfc, 4, 2, pci_host_data_writew, s);

    register_ioport_write(0xcfc, 4, 4, pci_host_data_writel, s);

    register_ioport_read(0xcfc, 4, 1, pci_host_data_readb, s);

    register_ioport_read(0xcfc, 4, 2, pci_host_data_readw, s);

    register_ioport_read(0xcfc, 4, 4, pci_host_data_readl, s);

    return 0;

}

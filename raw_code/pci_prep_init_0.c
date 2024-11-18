PCIBus *pci_prep_init(qemu_irq *pic)

{

    PREPPCIState *s;

    PCIDevice *d;

    int PPC_io_memory;



    s = qemu_mallocz(sizeof(PREPPCIState));

    s->bus = pci_register_bus(NULL, "pci",

                              prep_set_irq, prep_map_irq, pic, 0, 4);



    register_ioport_write(0xcf8, 4, 4, pci_prep_addr_writel, s);

    register_ioport_read(0xcf8, 4, 4, pci_prep_addr_readl, s);



    register_ioport_write(0xcfc, 4, 1, pci_host_data_writeb, s);

    register_ioport_write(0xcfc, 4, 2, pci_host_data_writew, s);

    register_ioport_write(0xcfc, 4, 4, pci_host_data_writel, s);

    register_ioport_read(0xcfc, 4, 1, pci_host_data_readb, s);

    register_ioport_read(0xcfc, 4, 2, pci_host_data_readw, s);

    register_ioport_read(0xcfc, 4, 4, pci_host_data_readl, s);



    PPC_io_memory = cpu_register_io_memory(PPC_PCIIO_read,

                                           PPC_PCIIO_write, s);

    cpu_register_physical_memory(0x80800000, 0x00400000, PPC_io_memory);



    /* PCI host bridge */

    d = pci_register_device(s->bus, "PREP Host Bridge - Motorola Raven",

                            sizeof(PCIDevice), 0, NULL, NULL);

    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_MOTOROLA);

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_MOTOROLA_RAVEN);

    d->config[0x08] = 0x00; // revision

    pci_config_set_class(d->config, PCI_CLASS_BRIDGE_HOST);

    d->config[0x0C] = 0x08; // cache_line_size

    d->config[0x0D] = 0x10; // latency_timer

    d->config[PCI_HEADER_TYPE] = PCI_HEADER_TYPE_NORMAL; // header_type

    d->config[0x34] = 0x00; // capabilities_pointer



    return s->bus;

}

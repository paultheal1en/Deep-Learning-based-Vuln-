PCIBus *pci_gt64120_init(qemu_irq *pic)

{

    GT64120State *s;

    PCIDevice *d;



    (void)&pci_host_data_writeb; /* avoid warning */

    (void)&pci_host_data_writew; /* avoid warning */

    (void)&pci_host_data_writel; /* avoid warning */

    (void)&pci_host_data_readb; /* avoid warning */

    (void)&pci_host_data_readw; /* avoid warning */

    (void)&pci_host_data_readl; /* avoid warning */



    s = qemu_mallocz(sizeof(GT64120State));

    s->pci = qemu_mallocz(sizeof(GT64120PCIState));



    s->pci->bus = pci_register_bus(NULL, "pci",

                                   pci_gt64120_set_irq, pci_gt64120_map_irq,

                                   pic, 144, 4);

    s->ISD_handle = cpu_register_io_memory(gt64120_read, gt64120_write, s);

    d = pci_register_device(s->pci->bus, "GT64120 PCI Bus", sizeof(PCIDevice),

                            0, gt64120_read_config, gt64120_write_config);



    /* FIXME: Malta specific hw assumptions ahead */



    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_MARVELL);

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_MARVELL_GT6412X);



    d->config[0x04] = 0x00;

    d->config[0x05] = 0x00;

    d->config[0x06] = 0x80;

    d->config[0x07] = 0x02;



    d->config[0x08] = 0x10;

    d->config[0x09] = 0x00;

    pci_config_set_class(d->config, PCI_CLASS_BRIDGE_HOST);



    d->config[0x10] = 0x08;

    d->config[0x14] = 0x08;

    d->config[0x17] = 0x01;

    d->config[0x1B] = 0x1c;

    d->config[0x1F] = 0x1f;

    d->config[0x23] = 0x14;

    d->config[0x24] = 0x01;

    d->config[0x27] = 0x14;

    d->config[0x3D] = 0x01;



    gt64120_reset(s);



    register_savevm("GT64120 PCI Bus", 0, 1, gt64120_save, gt64120_load, d);



    return s->pci->bus;

}

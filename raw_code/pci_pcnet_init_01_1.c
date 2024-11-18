PCIDevice *pci_pcnet_init(PCIBus *bus, NICInfo *nd, int devfn)

{

    PCNetState *d;

    uint8_t *pci_conf;



#if 0

    printf("sizeof(RMD)=%d, sizeof(TMD)=%d\n",

        sizeof(struct pcnet_RMD), sizeof(struct pcnet_TMD));

#endif



    d = (PCNetState *)pci_register_device(bus, "PCNet", sizeof(PCNetState),

                                          devfn, NULL, NULL);



    pci_conf = d->dev.config;



    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_AMD);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_AMD_LANCE);

    *(uint16_t *)&pci_conf[0x04] = cpu_to_le16(0x0007);

    *(uint16_t *)&pci_conf[0x06] = cpu_to_le16(0x0280);

    pci_conf[0x08] = 0x10;

    pci_conf[0x09] = 0x00;

    pci_config_set_class(pci_conf, PCI_CLASS_NETWORK_ETHERNET);

    pci_conf[0x0e] = 0x00; // header_type



    *(uint32_t *)&pci_conf[0x10] = cpu_to_le32(0x00000001);

    *(uint32_t *)&pci_conf[0x14] = cpu_to_le32(0x00000000);



    pci_conf[0x3d] = 1; // interrupt pin 0

    pci_conf[0x3e] = 0x06;

    pci_conf[0x3f] = 0xff;



    /* Handler for memory-mapped I/O */

    d->mmio_index =

      cpu_register_io_memory(0, pcnet_mmio_read, pcnet_mmio_write, d);



    pci_register_io_region((PCIDevice *)d, 0, PCNET_IOPORT_SIZE,

                           PCI_ADDRESS_SPACE_IO, pcnet_ioport_map);



    pci_register_io_region((PCIDevice *)d, 1, PCNET_PNPMMIO_SIZE,

                           PCI_ADDRESS_SPACE_MEM, pcnet_mmio_map);



    d->irq = d->dev.irq[0];

    d->phys_mem_read = pci_physical_memory_read;

    d->phys_mem_write = pci_physical_memory_write;

    d->pci_dev = &d->dev;



    pcnet_common_init(d, nd);

    return (PCIDevice *)d;

}

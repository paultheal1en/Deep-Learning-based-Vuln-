static int pci_pcnet_init(PCIDevice *pci_dev)

{

    PCIPCNetState *d = DO_UPCAST(PCIPCNetState, pci_dev, pci_dev);

    PCNetState *s = &d->state;

    uint8_t *pci_conf;



#if 0

    printf("sizeof(RMD)=%d, sizeof(TMD)=%d\n",

        sizeof(struct pcnet_RMD), sizeof(struct pcnet_TMD));

#endif



    pci_conf = pci_dev->config;



    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_AMD);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_AMD_LANCE);

    pci_set_word(pci_conf + PCI_STATUS,

                 PCI_STATUS_FAST_BACK | PCI_STATUS_DEVSEL_MEDIUM);

    pci_conf[PCI_REVISION_ID] = 0x10;

    pci_config_set_class(pci_conf, PCI_CLASS_NETWORK_ETHERNET);



    pci_set_word(pci_conf + PCI_SUBSYSTEM_VENDOR_ID, 0x0);

    pci_set_word(pci_conf + PCI_SUBSYSTEM_ID, 0x0);



    pci_conf[PCI_INTERRUPT_PIN] = 1; // interrupt pin 0

    pci_conf[PCI_MIN_GNT] = 0x06;

    pci_conf[PCI_MAX_LAT] = 0xff;



    /* Handler for memory-mapped I/O */

    s->mmio_index =

      cpu_register_io_memory(pcnet_mmio_read, pcnet_mmio_write, &d->state,

                             DEVICE_NATIVE_ENDIAN);



    pci_register_bar(pci_dev, 0, PCNET_IOPORT_SIZE,

                           PCI_BASE_ADDRESS_SPACE_IO, pcnet_ioport_map);



    pci_register_bar_simple(pci_dev, 1, PCNET_PNPMMIO_SIZE, 0, s->mmio_index);



    s->irq = pci_dev->irq[0];

    s->phys_mem_read = pci_physical_memory_read;

    s->phys_mem_write = pci_physical_memory_write;



    if (!pci_dev->qdev.hotplugged) {

        static int loaded = 0;

        if (!loaded) {

            rom_add_option("pxe-pcnet.rom", -1);

            loaded = 1;

        }

    }



    return pcnet_common_init(&pci_dev->qdev, s, &net_pci_pcnet_info);

}

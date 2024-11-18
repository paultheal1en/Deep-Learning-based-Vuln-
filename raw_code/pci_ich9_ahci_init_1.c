static int pci_ich9_ahci_init(PCIDevice *dev)

{

    struct AHCIPCIState *d;

    d = DO_UPCAST(struct AHCIPCIState, card, dev);



    pci_config_set_vendor_id(d->card.config, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(d->card.config, PCI_DEVICE_ID_INTEL_82801IR);



    pci_config_set_class(d->card.config, PCI_CLASS_STORAGE_SATA);

    pci_config_set_revision(d->card.config, 0x02);

    pci_config_set_prog_interface(d->card.config, AHCI_PROGMODE_MAJOR_REV_1);



    d->card.config[PCI_CACHE_LINE_SIZE] = 0x08;  /* Cache line size */

    d->card.config[PCI_LATENCY_TIMER]   = 0x00;  /* Latency timer */

    pci_config_set_interrupt_pin(d->card.config, 1);



    /* XXX Software should program this register */

    d->card.config[0x90]   = 1 << 6; /* Address Map Register - AHCI mode */



    qemu_register_reset(ahci_reset, d);



    /* XXX BAR size should be 1k, but that breaks, so bump it to 4k for now */

    pci_register_bar_simple(&d->card, 5, 0x1000, 0, d->ahci.mem);



    msi_init(dev, 0x50, 1, true, false);



    ahci_init(&d->ahci, &dev->qdev, 6);

    d->ahci.irq = d->card.irq[0];



    return 0;

}

static int pci_cmd646_ide_initfn(PCIDevice *dev)

{

    PCIIDEState *d = DO_UPCAST(PCIIDEState, dev, dev);

    uint8_t *pci_conf = d->dev.config;

    qemu_irq *irq;

    int i;



    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_CMD);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_CMD_646);



    pci_conf[PCI_REVISION_ID] = 0x07; // IDE controller revision

    pci_conf[PCI_CLASS_PROG] = 0x8f;



    pci_config_set_class(pci_conf, PCI_CLASS_STORAGE_IDE);



    pci_conf[0x51] = 0x04; // enable IDE0

    if (d->secondary) {

        /* XXX: if not enabled, really disable the seconday IDE controller */

        pci_conf[0x51] |= 0x08; /* enable IDE1 */

    }



    pci_register_bar(dev, 0, 0x8, PCI_BASE_ADDRESS_SPACE_IO, ide_map);

    pci_register_bar(dev, 1, 0x4, PCI_BASE_ADDRESS_SPACE_IO, ide_map);

    pci_register_bar(dev, 2, 0x8, PCI_BASE_ADDRESS_SPACE_IO, ide_map);

    pci_register_bar(dev, 3, 0x4, PCI_BASE_ADDRESS_SPACE_IO, ide_map);

    pci_register_bar(dev, 4, 0x10, PCI_BASE_ADDRESS_SPACE_IO, bmdma_map);



    /* TODO: RST# value should be 0 */

    pci_conf[PCI_INTERRUPT_PIN] = 0x01; // interrupt on pin 1



    irq = qemu_allocate_irqs(cmd646_set_irq, d, 2);

    for (i = 0; i < 2; i++) {

        ide_bus_new(&d->bus[i], &d->dev.qdev, i);

        ide_init2(&d->bus[i], irq[i]);



        bmdma_init(&d->bus[i], &d->bmdma[i]);

        d->bmdma[i].bus = &d->bus[i];

        qemu_add_vm_change_state_handler(d->bus[i].dma->ops->restart_cb,

                                         &d->bmdma[i].dma);

    }



    vmstate_register(&dev->qdev, 0, &vmstate_ide_pci, d);

    qemu_register_reset(cmd646_reset, d);

    return 0;

}

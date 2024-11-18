static int usb_ohci_initfn_pci(struct PCIDevice *dev)

{

    OHCIPCIState *ohci = DO_UPCAST(OHCIPCIState, pci_dev, dev);

    int num_ports = 3;



    pci_config_set_vendor_id(ohci->pci_dev.config, PCI_VENDOR_ID_APPLE);

    pci_config_set_device_id(ohci->pci_dev.config,

                             PCI_DEVICE_ID_APPLE_IPID_USB);

    ohci->pci_dev.config[PCI_CLASS_PROG] = 0x10; /* OHCI */

    pci_config_set_class(ohci->pci_dev.config, PCI_CLASS_SERIAL_USB);

    /* TODO: RST# value should be 0. */

    ohci->pci_dev.config[PCI_INTERRUPT_PIN] = 0x01; /* interrupt pin 1 */



    usb_ohci_init(&ohci->state, &dev->qdev, num_ports, 0);

    ohci->state.irq = ohci->pci_dev.irq[0];



    /* TODO: avoid cast below by using dev */

    pci_register_bar_simple(&ohci->pci_dev, 0, 256, 0, ohci->state.mem);

    return 0;

}

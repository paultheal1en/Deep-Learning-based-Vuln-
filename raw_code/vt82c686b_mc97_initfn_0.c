static int vt82c686b_mc97_initfn(PCIDevice *dev)

{

    VT686MC97State *s = DO_UPCAST(VT686MC97State, dev, dev);

    uint8_t *pci_conf = s->dev.config;



    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_VIA);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_VIA_MC97);

    pci_config_set_class(pci_conf, PCI_CLASS_COMMUNICATION_OTHER);

    pci_config_set_revision(pci_conf, 0x30);



    pci_set_word(pci_conf + PCI_COMMAND, PCI_COMMAND_INVALIDATE |

                 PCI_COMMAND_VGA_PALETTE);

    pci_set_word(pci_conf + PCI_STATUS, PCI_STATUS_DEVSEL_MEDIUM);

    pci_set_long(pci_conf + PCI_INTERRUPT_PIN, 0x03);



    return 0;

}

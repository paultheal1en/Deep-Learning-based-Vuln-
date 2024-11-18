static int pbm_pci_host_init(PCIDevice *d)

{

    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_SUN);

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_SUN_SABRE);

    pci_set_word(d->config + PCI_COMMAND,

                 PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER);

    pci_set_word(d->config + PCI_STATUS,

                 PCI_STATUS_FAST_BACK | PCI_STATUS_66MHZ |

                 PCI_STATUS_DEVSEL_MEDIUM);

    pci_config_set_class(d->config, PCI_CLASS_BRIDGE_HOST);

    return 0;

}

static void pci_init_wmask(PCIDevice *dev)

{

    int i;

    int config_size = pci_config_size(dev);



    dev->wmask[PCI_CACHE_LINE_SIZE] = 0xff;

    dev->wmask[PCI_INTERRUPT_LINE] = 0xff;

    pci_set_word(dev->wmask + PCI_COMMAND,

                 PCI_COMMAND_IO | PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER);

    for (i = PCI_CONFIG_HEADER_SIZE; i < config_size; ++i)

        dev->wmask[i] = 0xff;

}

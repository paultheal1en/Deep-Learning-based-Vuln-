static void apb_pci_bridge_init(PCIBus *b)

{

    PCIDevice *dev = pci_bridge_get_device(b);



    /*

     * command register:

     * According to PCI bridge spec, after reset

     *   bus master bit is off

     *   memory space enable bit is off

     * According to manual (805-1251.pdf).

     *   the reset value should be zero unless the boot pin is tied high

     *   (which is true) and thus it should be PCI_COMMAND_MEMORY.

     */

    pci_set_word(dev->config + PCI_COMMAND,

                 PCI_COMMAND_MEMORY);

    pci_set_word(dev->config + PCI_STATUS,

                 PCI_STATUS_FAST_BACK | PCI_STATUS_66MHZ |

                 PCI_STATUS_DEVSEL_MEDIUM);

    pci_set_byte(dev->config + PCI_REVISION_ID, 0x11);

    pci_set_byte(dev->config + PCI_HEADER_TYPE,

                 pci_get_byte(dev->config + PCI_HEADER_TYPE) |

                 PCI_HEADER_TYPE_MULTI_FUNCTION);

}

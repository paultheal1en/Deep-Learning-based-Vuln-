static int apb_pci_bridge_initfn(PCIDevice *dev)

{

    int rc;



    rc = pci_bridge_initfn(dev);

    if (rc < 0) {

        return rc;

    }



    pci_config_set_vendor_id(dev->config, PCI_VENDOR_ID_SUN);

    pci_config_set_device_id(dev->config, PCI_DEVICE_ID_SUN_SIMBA);



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

    return 0;

}

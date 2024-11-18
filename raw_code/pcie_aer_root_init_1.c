void pcie_aer_root_init(PCIDevice *dev)

{

    uint16_t pos = dev->exp.aer_cap;



    pci_set_long(dev->wmask + pos + PCI_ERR_ROOT_COMMAND,

                 PCI_ERR_ROOT_CMD_EN_MASK);

    pci_set_long(dev->w1cmask + pos + PCI_ERR_ROOT_STATUS,

                 PCI_ERR_ROOT_STATUS_REPORT_MASK);






}
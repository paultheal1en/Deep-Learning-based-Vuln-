PCIDevice *pci_get_function_0(PCIDevice *pci_dev)

{

    if(pcie_has_upstream_port(pci_dev)) {

        /* With an upstream PCIe port, we only support 1 device at slot 0 */

        return pci_dev->bus->devices[0];

    } else {

        /* Other bus types might support multiple devices at slots 0-31 */

        return pci_dev->bus->devices[PCI_DEVFN(PCI_SLOT(pci_dev->devfn), 0)];

    }

}

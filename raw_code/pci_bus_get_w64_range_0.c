void pci_bus_get_w64_range(PCIBus *bus, Range *range)

{

    range->begin = range->end = 0;

    pci_for_each_device_under_bus(bus, pci_dev_get_w64, range);

}

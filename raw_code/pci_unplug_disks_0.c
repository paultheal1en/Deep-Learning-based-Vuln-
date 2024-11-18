static void pci_unplug_disks(PCIBus *bus)

{

    pci_for_each_device(bus, 0, unplug_disks, NULL);

}

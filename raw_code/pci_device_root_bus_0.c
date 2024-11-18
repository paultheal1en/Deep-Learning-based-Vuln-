PCIBus *pci_device_root_bus(const PCIDevice *d)

{

    PCIBus *bus = d->bus;



    while (!pci_bus_is_root(bus)) {

        d = bus->parent_dev;

        assert(d != NULL);



        bus = d->bus;

    }



    return bus;

}

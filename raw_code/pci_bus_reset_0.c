void pci_bus_reset(PCIBus *bus)

{

    int i;



    for (i = 0; i < bus->nirq; i++) {

        bus->irq_count[i] = 0;

    }

    for (i = 0; i < ARRAY_SIZE(bus->devices); ++i) {

        if (bus->devices[i]) {

            pci_device_reset(bus->devices[i]);

        }

    }

}

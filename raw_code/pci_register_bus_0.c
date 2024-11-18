PCIBus *pci_register_bus(DeviceState *parent, const char *name,

                         pci_set_irq_fn set_irq, pci_map_irq_fn map_irq,

                         qemu_irq *pic, int devfn_min, int nirq)

{

    PCIBus *bus;

    static int nbus = 0;



    bus = FROM_QBUS(PCIBus, qbus_create(&pci_bus_info, parent, name));

    bus->set_irq = set_irq;

    bus->map_irq = map_irq;

    bus->irq_opaque = pic;

    bus->devfn_min = devfn_min;

    bus->nirq = nirq;

    bus->irq_count = qemu_malloc(nirq * sizeof(bus->irq_count[0]));

    bus->next = first_bus;

    first_bus = bus;

    register_savevm("PCIBUS", nbus++, 1, pcibus_save, pcibus_load, bus);

    qemu_register_reset(pci_bus_reset, bus);

    return bus;

}

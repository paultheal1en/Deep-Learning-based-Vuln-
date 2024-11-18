static void pci_set_irq(void *opaque, int irq_num, int level)

{

    PCIDevice *pci_dev = opaque;

    PCIBus *bus;

    int change;



    change = level - pci_dev->irq_state[irq_num];

    if (!change)

        return;



    pci_dev->irq_state[irq_num] = level;

    for (;;) {

        bus = pci_dev->bus;

        irq_num = bus->map_irq(pci_dev, irq_num);

        if (bus->set_irq)

            break;

        pci_dev = bus->parent_dev;

    }

    bus->irq_count[irq_num] += change;

    bus->set_irq(bus->irq_opaque, irq_num, bus->irq_count[irq_num] != 0);

}

PCIINTxRoute pci_device_route_intx_to_irq(PCIDevice *dev, int pin)

{

    PCIBus *bus;



    do {

         bus = dev->bus;

         pin = bus->map_irq(dev, pin);

         dev = bus->parent_dev;

    } while (dev);

    assert(bus->route_intx_to_irq);

    return bus->route_intx_to_irq(bus->irq_opaque, pin);

}

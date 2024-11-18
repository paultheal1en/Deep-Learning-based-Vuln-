PCIBus *pci_bridge_init(PCIBus *bus, int devfn, uint16_t vid, uint16_t did,

                        pci_map_irq_fn map_irq, const char *name)

{

    PCIDevice *dev;

    PCIBridge *s;



    dev = pci_create(bus, devfn, "pci-bridge");

    qdev_prop_set_uint32(&dev->qdev, "vendorid", vid);

    qdev_prop_set_uint32(&dev->qdev, "deviceid", did);

    qdev_init(&dev->qdev);



    s = DO_UPCAST(PCIBridge, dev, dev);

    pci_register_secondary_bus(&s->bus, &s->dev, map_irq, name);

    return &s->bus;

}

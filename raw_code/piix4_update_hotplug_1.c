static void piix4_update_hotplug(PIIX4PMState *s)

{

    PCIDevice *dev = &s->dev;

    BusState *bus = qdev_get_parent_bus(&dev->qdev);

    DeviceState *qdev, *next;



    s->pci0_hotplug_enable = ~0;



    QTAILQ_FOREACH_SAFE(qdev, &bus->children, sibling, next) {

        PCIDevice *pdev = PCI_DEVICE(qdev);

        PCIDeviceClass *pc = PCI_DEVICE_GET_CLASS(pdev);

        int slot = PCI_SLOT(pdev->devfn);



        if (pc->no_hotplug) {

            s->pci0_hotplug_enable &= ~(1 << slot);

        }

    }

}

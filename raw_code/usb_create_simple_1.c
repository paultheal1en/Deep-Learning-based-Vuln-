USBDevice *usb_create_simple(USBBus *bus, const char *name)

{

    USBDevice *dev = usb_create(bus, name);

    qdev_init(&dev->qdev);

    return dev;

}

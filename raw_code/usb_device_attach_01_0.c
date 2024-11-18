int usb_device_attach(USBDevice *dev)

{

    USBBus *bus = usb_bus_from_device(dev);



    if (bus->nfree == 1 && dev->port_path == NULL) {

        /* Create a new hub and chain it on

           (unless a physical port location is specified). */

        usb_create_simple(bus, "usb-hub");

    }

    return do_attach(dev);

}

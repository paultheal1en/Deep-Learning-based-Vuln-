int usb_claim_port(USBDevice *dev)

{

    USBBus *bus = usb_bus_from_device(dev);

    USBPort *port;



    assert(dev->port == NULL);



    if (dev->port_path) {

        QTAILQ_FOREACH(port, &bus->free, next) {

            if (strcmp(port->path, dev->port_path) == 0) {

                break;

            }

        }

        if (port == NULL) {

            error_report("Error: usb port %s (bus %s) not found (in use?)",

                         dev->port_path, bus->qbus.name);

            return -1;

        }

    } else {

        if (bus->nfree == 1 && strcmp(object_get_typename(OBJECT(dev)), "usb-hub") != 0) {

            /* Create a new hub and chain it on */

            usb_create_simple(bus, "usb-hub");

        }

        if (bus->nfree == 0) {

            error_report("Error: tried to attach usb device %s to a bus "

                         "with no free ports", dev->product_desc);

            return -1;

        }

        port = QTAILQ_FIRST(&bus->free);

    }

    trace_usb_port_claim(bus->busnr, port->path);



    QTAILQ_REMOVE(&bus->free, port, next);

    bus->nfree--;



    dev->port = port;

    port->dev = dev;



    QTAILQ_INSERT_TAIL(&bus->used, port, next);

    bus->nused++;

    return 0;

}

static int do_attach(USBDevice *dev)

{

    USBBus *bus = usb_bus_from_device(dev);

    USBPort *port;



    if (dev->attached) {

        error_report("Error: tried to attach usb device %s twice\n",

                dev->product_desc);

        return -1;

    }

    if (bus->nfree == 0) {

        error_report("Error: tried to attach usb device %s to a bus with no free ports\n",

                dev->product_desc);

        return -1;

    }

    if (dev->port_path) {

        QTAILQ_FOREACH(port, &bus->free, next) {

            if (strcmp(port->path, dev->port_path) == 0) {

                break;

            }

        }

        if (port == NULL) {

            error_report("Error: usb port %s (bus %s) not found\n",

                    dev->port_path, bus->qbus.name);

            return -1;

        }

    } else {

        port = QTAILQ_FIRST(&bus->free);

    }

    if (!(port->speedmask & dev->speedmask)) {

        error_report("Warning: speed mismatch trying to attach usb device %s to bus %s\n",

                dev->product_desc, bus->qbus.name);

        return -1;

    }



    dev->attached++;

    QTAILQ_REMOVE(&bus->free, port, next);

    bus->nfree--;



    usb_attach(port, dev);



    QTAILQ_INSERT_TAIL(&bus->used, port, next);

    bus->nused++;



    return 0;

}

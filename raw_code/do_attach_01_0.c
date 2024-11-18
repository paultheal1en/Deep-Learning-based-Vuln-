static void do_attach(USBDevice *dev)

{

    USBBus *bus = usb_bus_from_device(dev);

    USBPort *port;



    if (dev->attached) {

        fprintf(stderr, "Warning: tried to attach usb device %s twice\n",

                dev->devname);

        return;

    }

    dev->attached++;



    port = TAILQ_FIRST(&bus->free);

    TAILQ_REMOVE(&bus->free, port, next);

    bus->nfree--;



    usb_attach(port, dev);



    TAILQ_INSERT_TAIL(&bus->used, port, next);

    bus->nused++;

}

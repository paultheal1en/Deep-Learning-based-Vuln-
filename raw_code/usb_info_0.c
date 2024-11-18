void usb_info(Monitor *mon)

{

    USBBus *bus;

    USBDevice *dev;

    USBPort *port;



    if (TAILQ_EMPTY(&busses)) {

        monitor_printf(mon, "USB support not enabled\n");

        return;

    }



    TAILQ_FOREACH(bus, &busses, next) {

        TAILQ_FOREACH(port, &bus->used, next) {

            dev = port->dev;

            if (!dev)

                continue;

            monitor_printf(mon, "  Device %d.%d, Speed %s Mb/s, Product %s\n",

                           bus->busnr, dev->addr, usb_speed(dev->speed), dev->devname);

        }

    }

}

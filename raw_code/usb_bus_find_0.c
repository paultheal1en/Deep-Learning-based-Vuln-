USBBus *usb_bus_find(int busnr)

{

    USBBus *bus;



    if (-1 == busnr)

        return TAILQ_FIRST(&busses);

    TAILQ_FOREACH(bus, &busses, next) {

        if (bus->busnr == busnr)

            return bus;

    }

    return NULL;

}

static int usb_host_auto_scan(void *opaque, int bus_num, int addr,

                     int class_id, int vendor_id, int product_id,

                     const char *product_name, int speed)

{

    struct USBAutoFilter *f;

    struct USBDevice *dev;



    /* Ignore hubs */

    if (class_id == 9)

        return 0;



    for (f = usb_auto_filter; f; f = f->next) {

	if (f->bus_num >= 0 && f->bus_num != bus_num)

            continue;



	if (f->addr >= 0 && f->addr != addr)

            continue;



	if (f->vendor_id >= 0 && f->vendor_id != vendor_id)

            continue;



	if (f->product_id >= 0 && f->product_id != product_id)

            continue;



        /* We got a match */



        /* Allredy attached ? */

        if (hostdev_find(bus_num, addr))

            return 0;



        dprintf("husb: auto open: bus_num %d addr %d\n", bus_num, addr);



	dev = usb_host_device_open_addr(bus_num, addr, product_name);

    }



    return 0;

}

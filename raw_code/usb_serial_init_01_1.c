static USBDevice *usb_serial_init(USBBus *bus, const char *filename)

{

    USBDevice *dev;

    CharDriverState *cdrv;

    uint32_t vendorid = 0, productid = 0;

    char label[32];

    static int index;



    while (*filename && *filename != ':') {

        const char *p;

        char *e;

        if (strstart(filename, "vendorid=", &p)) {

            vendorid = strtol(p, &e, 16);

            if (e == p || (*e && *e != ',' && *e != ':')) {

                error_report("bogus vendor ID %s", p);

                return NULL;

            }

            filename = e;

        } else if (strstart(filename, "productid=", &p)) {

            productid = strtol(p, &e, 16);

            if (e == p || (*e && *e != ',' && *e != ':')) {

                error_report("bogus product ID %s", p);

                return NULL;

            }

            filename = e;

        } else {

            error_report("unrecognized serial USB option %s", filename);

            return NULL;

        }

        while(*filename == ',')

            filename++;

    }

    if (!*filename) {

        error_report("character device specification needed");

        return NULL;

    }

    filename++;



    snprintf(label, sizeof(label), "usbserial%d", index++);

    cdrv = qemu_chr_new(label, filename, NULL);

    if (!cdrv)

        return NULL;



    dev = usb_create(bus, "usb-serial");

    qdev_prop_set_chr(&dev->qdev, "chardev", cdrv);

    if (vendorid)

        qdev_prop_set_uint16(&dev->qdev, "vendorid", vendorid);

    if (productid)

        qdev_prop_set_uint16(&dev->qdev, "productid", productid);

    qdev_init_nofail(&dev->qdev);



    return dev;

}

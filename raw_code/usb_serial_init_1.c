static USBDevice *usb_serial_init(const char *filename)

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

                qemu_error("bogus vendor ID %s\n", p);



            filename = e;

        } else if (strstart(filename, "productid=", &p)) {

            productid = strtol(p, &e, 16);

            if (e == p || (*e && *e != ',' && *e != ':')) {

                qemu_error("bogus product ID %s\n", p);



            filename = e;

        } else {

            qemu_error("unrecognized serial USB option %s\n", filename);



        while(*filename == ',')

            filename++;


    if (!*filename) {

        qemu_error("character device specification needed\n");



    filename++;



    snprintf(label, sizeof(label), "usbserial%d", index++);

    cdrv = qemu_chr_open(label, filename, NULL);

    if (!cdrv)




    dev = usb_create(NULL /* FIXME */, "usb-serial");




    qdev_prop_set_chr(&dev->qdev, "chardev", cdrv);

    if (vendorid)

        qdev_prop_set_uint16(&dev->qdev, "vendorid", vendorid);

    if (productid)

        qdev_prop_set_uint16(&dev->qdev, "productid", productid);

    qdev_init_nofail(&dev->qdev);



    return dev;

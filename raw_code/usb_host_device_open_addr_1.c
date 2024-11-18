static USBDevice *usb_host_device_open_addr(int bus_num, int addr, const char *prod_name)

{

    int fd = -1, ret;

    USBDevice *d = NULL;

    USBHostDevice *dev;

    struct usbdevfs_connectinfo ci;

    char buf[1024];



    printf("husb: open device %d.%d\n", bus_num, addr);



    if (!usb_host_device_path) {

        perror("husb: USB Host Device Path not set");

        goto fail;

    }

    snprintf(buf, sizeof(buf), "%s/%03d/%03d", usb_host_device_path,

             bus_num, addr);

    fd = open(buf, O_RDWR | O_NONBLOCK);

    if (fd < 0) {

        perror(buf);

        goto fail;

    }

    dprintf("husb: opened %s\n", buf);



    d = usb_create(NULL /* FIXME */, "USB Host Device");

    dev = DO_UPCAST(USBHostDevice, dev, d);



    dev->bus_num = bus_num;

    dev->addr = addr;

    dev->fd = fd;



    /* read the device description */

    dev->descr_len = read(fd, dev->descr, sizeof(dev->descr));

    if (dev->descr_len <= 0) {

        perror("husb: reading device data failed");

        goto fail;

    }



#ifdef DEBUG

    {

        int x;

        printf("=== begin dumping device descriptor data ===\n");

        for (x = 0; x < dev->descr_len; x++)

            printf("%02x ", dev->descr[x]);

        printf("\n=== end dumping device descriptor data ===\n");

    }

#endif





    /* 

     * Initial configuration is -1 which makes us claim first 

     * available config. We used to start with 1, which does not

     * always work. I've seen devices where first config starts 

     * with 2.

     */

    if (!usb_host_claim_interfaces(dev, -1))

        goto fail;



    ret = ioctl(fd, USBDEVFS_CONNECTINFO, &ci);

    if (ret < 0) {

        perror("usb_host_device_open: USBDEVFS_CONNECTINFO");

        goto fail;

    }



    printf("husb: grabbed usb device %d.%d\n", bus_num, addr);



    ret = usb_linux_update_endp_table(dev);

    if (ret)

        goto fail;



    if (ci.slow)

        dev->dev.speed = USB_SPEED_LOW;

    else

        dev->dev.speed = USB_SPEED_HIGH;



    if (!prod_name || prod_name[0] == '\0')

        snprintf(dev->dev.devname, sizeof(dev->dev.devname),

                 "host:%d.%d", bus_num, addr);

    else

        pstrcpy(dev->dev.devname, sizeof(dev->dev.devname),

                prod_name);



    /* USB devio uses 'write' flag to check for async completions */

    qemu_set_fd_handler(dev->fd, NULL, async_complete, dev);



    hostdev_link(dev);



    qdev_init(&d->qdev);

    return (USBDevice *) dev;



fail:

    if (d)

        qdev_free(&d->qdev);

    if (fd != -1)

        close(fd);

    return NULL;

}

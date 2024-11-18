USBDevice *usb_host_device_open(const char *devname)

{

    struct usb_device_info bus_info, dev_info;

    USBDevice *d = NULL;

    USBHostDevice *dev;

    char ctlpath[PATH_MAX + 1];

    char buspath[PATH_MAX + 1];

    int bfd, dfd, bus, address, i;

    int ugendebug = UGEN_DEBUG_LEVEL;



    if (usb_host_find_device(&bus, &address, devname) < 0)

        return NULL;



    snprintf(buspath, PATH_MAX, "/dev/usb%d", bus);



    bfd = open(buspath, O_RDWR);

    if (bfd < 0) {

#ifdef DEBUG

        printf("usb_host_device_open: failed to open usb bus - %s\n",

               strerror(errno));

#endif

        return NULL;

    }



    bus_info.udi_addr = address;

    if (ioctl(bfd, USB_DEVICEINFO, &bus_info) < 0) {

#ifdef DEBUG

        printf("usb_host_device_open: failed to grab bus information - %s\n",

               strerror(errno));

#endif

        return NULL;

    }



#if defined(__FreeBSD__) || defined(__FreeBSD_kernel__) || defined(__DragonFly__)

    snprintf(ctlpath, PATH_MAX, "/dev/%s", bus_info.udi_devnames[0]);

#else

    snprintf(ctlpath, PATH_MAX, "/dev/%s.00", bus_info.udi_devnames[0]);

#endif



    dfd  = open(ctlpath, O_RDWR);

    if (dfd < 0) {

        dfd = open(ctlpath, O_RDONLY);

        if (dfd < 0) {

#ifdef DEBUG

            printf("usb_host_device_open: failed to open usb device %s - %s\n",

                   ctlpath, strerror(errno));

#endif

        }

    }



    if (dfd >= 0) {

        if (ioctl(dfd, USB_GET_DEVICEINFO, &dev_info) < 0) {

#ifdef DEBUG

            printf("usb_host_device_open: failed to grab device info - %s\n",

                   strerror(errno));

#endif

            goto fail;

        }



        d = usb_create(NULL /* FIXME */, "usb-host");

        dev = DO_UPCAST(USBHostDevice, dev, d);



        if (dev_info.udi_speed == 1)

            dev->dev.speed = USB_SPEED_LOW - 1;

        else

            dev->dev.speed = USB_SPEED_FULL - 1;



        if (strncmp(dev_info.udi_product, "product", 7) != 0)

            pstrcpy(dev->dev.product_desc, sizeof(dev->dev.product_desc),

                    dev_info.udi_product);

        else

            snprintf(dev->dev.product_desc, sizeof(dev->dev.product_desc),

                     "host:%s", devname);



        pstrcpy(dev->devpath, sizeof(dev->devpath), "/dev/");

        pstrcat(dev->devpath, sizeof(dev->devpath), dev_info.udi_devnames[0]);



        /* Mark the endpoints as not yet open */

        for (i = 0; i < USB_MAX_ENDPOINTS; i++)

           dev->ep_fd[i] = -1;



        ioctl(dfd, USB_SETDEBUG, &ugendebug);



        return (USBDevice *)dev;

    }



fail:

    return NULL;

}

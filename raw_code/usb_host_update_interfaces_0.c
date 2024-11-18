static int usb_host_update_interfaces(USBHostDevice *dev, int configuration)

{

    int dev_descr_len, config_descr_len;

    int interface, nb_interfaces, nb_configurations;

    int ret, i;



    if (configuration == 0) /* address state - ignore */

        return 1;



    i = 0;

    dev_descr_len = dev->descr[0];

    if (dev_descr_len > dev->descr_len)

        goto fail;

    nb_configurations = dev->descr[17];



    i += dev_descr_len;

    while (i < dev->descr_len) {

#ifdef DEBUG

        printf("i is %d, descr_len is %d, dl %d, dt %d\n", i, dev->descr_len,

               dev->descr[i], dev->descr[i+1]);

#endif

        if (dev->descr[i+1] != USB_DT_CONFIG) {

            i += dev->descr[i];

            continue;

        }

        config_descr_len = dev->descr[i];



        if (configuration == dev->descr[i + 5])

            break;



        i += config_descr_len;

    }



    if (i >= dev->descr_len) {

        printf("usb_host: error - device has no matching configuration\n");

        goto fail;

    }

    nb_interfaces = dev->descr[i + 4];



#ifdef USBDEVFS_DISCONNECT

    /* earlier Linux 2.4 do not support that */

    {

        struct usbdevfs_ioctl ctrl;

        for (interface = 0; interface < nb_interfaces; interface++) {

            ctrl.ioctl_code = USBDEVFS_DISCONNECT;

            ctrl.ifno = interface;

            ret = ioctl(dev->fd, USBDEVFS_IOCTL, &ctrl);

            if (ret < 0 && errno != ENODATA) {

                perror("USBDEVFS_DISCONNECT");

                goto fail;

            }

        }

    }

#endif



    /* XXX: only grab if all interfaces are free */

    for (interface = 0; interface < nb_interfaces; interface++) {

        ret = ioctl(dev->fd, USBDEVFS_CLAIMINTERFACE, &interface);

        if (ret < 0) {

            if (errno == EBUSY) {

                fprintf(stderr,

                        "usb_host: warning - device already grabbed\n");

            } else {

                perror("USBDEVFS_CLAIMINTERFACE");

            }

        fail:

            return 0;

        }

    }



#ifdef DEBUG

    printf("usb_host: %d interfaces claimed for configuration %d\n",

           nb_interfaces, configuration);

#endif



    return 1;

}

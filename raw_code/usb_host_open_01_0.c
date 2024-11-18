static int usb_host_open(USBHostDevice *s, libusb_device *dev)

{

    USBDevice *udev = USB_DEVICE(s);

    int bus_num = libusb_get_bus_number(dev);

    int addr    = libusb_get_device_address(dev);

    int rc;



    trace_usb_host_open_started(bus_num, addr);



    if (s->dh != NULL) {

        goto fail;

    }

    rc = libusb_open(dev, &s->dh);

    if (rc != 0) {

        goto fail;

    }



    s->dev     = dev;

    s->bus_num = bus_num;

    s->addr    = addr;



    usb_host_detach_kernel(s);



    libusb_get_device_descriptor(dev, &s->ddesc);

    usb_host_get_port(s->dev, s->port, sizeof(s->port));



    usb_ep_init(udev);

    usb_host_ep_update(s);



    udev->speed     = speed_map[libusb_get_device_speed(dev)];

    usb_host_speed_compat(s);



    if (s->ddesc.iProduct) {

        libusb_get_string_descriptor_ascii(s->dh, s->ddesc.iProduct,

                                           (unsigned char *)udev->product_desc,

                                           sizeof(udev->product_desc));

    } else {

        snprintf(udev->product_desc, sizeof(udev->product_desc),

                 "host:%d.%d", bus_num, addr);

    }



    rc = usb_device_attach(udev);

    if (rc) {

        goto fail;

    }



    trace_usb_host_open_success(bus_num, addr);

    return 0;



fail:

    trace_usb_host_open_failure(bus_num, addr);

    if (s->dh != NULL) {

        libusb_close(s->dh);

        s->dh = NULL;

        s->dev = NULL;

    }

    return -1;

}

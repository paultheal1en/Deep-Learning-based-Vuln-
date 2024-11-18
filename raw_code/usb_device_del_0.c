static int usb_device_del(const char *devname)

{

    USBPort *port;

    USBPort **lastp;

    USBDevice *dev;

    int bus_num, addr;

    const char *p;



    if (!used_usb_ports)

        return -1;



    p = strchr(devname, '.');

    if (!p)

        return -1;

    bus_num = strtoul(devname, NULL, 0);

    addr = strtoul(p + 1, NULL, 0);

    if (bus_num != 0)

        return -1;



    lastp = &used_usb_ports;

    port = used_usb_ports;

    while (port && port->dev->addr != addr) {

        lastp = &port->next;

        port = port->next;

    }



    if (!port)

        return -1;



    dev = port->dev;

    *lastp = port->next;

    usb_attach(port, NULL);

    dev->handle_destroy(dev);

    port->next = free_usb_ports;

    free_usb_ports = port;

    return 0;

}

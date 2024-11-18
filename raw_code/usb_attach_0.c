void usb_attach(USBPort *port, USBDevice *dev)

{

    if (dev != NULL) {

        /* attach */

        if (port->dev) {

            usb_attach(port, NULL);

        }

        dev->port = port;

        port->dev = dev;

        port->ops->attach(port);

        usb_send_msg(dev, USB_MSG_ATTACH);

    } else {

        /* detach */

        dev = port->dev;

        port->ops->detach(port);

        if (dev) {

            usb_send_msg(dev, USB_MSG_DETACH);

            dev->port = NULL;

            port->dev = NULL;

        }

    }

}

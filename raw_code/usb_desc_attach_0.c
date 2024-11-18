void usb_desc_attach(USBDevice *dev)

{

    const USBDesc *desc = usb_device_get_usb_desc(dev);



    assert(desc != NULL);

    if (desc->super && (dev->port->speedmask & USB_SPEED_MASK_SUPER)) {

        dev->speed = USB_SPEED_SUPER;

    } else if (desc->high && (dev->port->speedmask & USB_SPEED_MASK_HIGH)) {

        dev->speed = USB_SPEED_HIGH;

    } else if (desc->full && (dev->port->speedmask & USB_SPEED_MASK_FULL)) {

        dev->speed = USB_SPEED_FULL;

    } else {

        return;

    }

    usb_desc_setdefaults(dev);

}

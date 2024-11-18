static void usb_serial_realize(USBDevice *dev, Error **errp)

{

    USBSerialState *s = DO_UPCAST(USBSerialState, dev, dev);



    usb_desc_create_serial(dev);

    usb_desc_init(dev);

    dev->auto_attach = 0;



    if (!s->cs) {

        error_setg(errp, "Property chardev is required");

        return;

    }



    qemu_chr_add_handlers(s->cs, usb_serial_can_read, usb_serial_read,

                          usb_serial_event, s);

    usb_serial_handle_reset(dev);



    if (s->cs->be_open && !dev->attached) {

        usb_device_attach(dev, errp);

    }

}

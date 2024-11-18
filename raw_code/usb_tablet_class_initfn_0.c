static void usb_tablet_class_initfn(ObjectClass *klass, void *data)

{

    USBDeviceClass *uc = USB_DEVICE_CLASS(klass);



    uc->init           = usb_tablet_initfn;

    uc->product_desc   = "QEMU USB Tablet";

    uc->usb_desc       = &desc_tablet;

    uc->handle_packet  = usb_generic_handle_packet;

    uc->handle_reset   = usb_hid_handle_reset;

    uc->handle_control = usb_hid_handle_control;

    uc->handle_data    = usb_hid_handle_data;

    uc->handle_destroy = usb_hid_handle_destroy;

}

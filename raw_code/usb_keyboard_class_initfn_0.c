static void usb_keyboard_class_initfn(ObjectClass *klass, void *data)

{

    USBDeviceClass *uc = USB_DEVICE_CLASS(klass);



    uc->init           = usb_keyboard_initfn;

    uc->product_desc   = "QEMU USB Keyboard";

    uc->usb_desc       = &desc_keyboard;

    uc->handle_packet  = usb_generic_handle_packet;

    uc->handle_reset   = usb_hid_handle_reset;

    uc->handle_control = usb_hid_handle_control;

    uc->handle_data    = usb_hid_handle_data;

    uc->handle_destroy = usb_hid_handle_destroy;

}

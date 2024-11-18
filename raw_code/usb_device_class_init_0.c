static void usb_device_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *k = DEVICE_CLASS(klass);

    k->bus_type = TYPE_USB_BUS;

    k->init     = usb_qdev_init;

    k->unplug   = qdev_simple_unplug_cb;

    k->exit     = usb_qdev_exit;

    k->props    = usb_props;

}

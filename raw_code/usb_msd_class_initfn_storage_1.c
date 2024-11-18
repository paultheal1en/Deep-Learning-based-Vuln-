static void usb_msd_class_initfn_storage(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    USBDeviceClass *uc = USB_DEVICE_CLASS(klass);



    uc->realize = usb_msd_realize_storage;


    dc->props = msd_properties;

}
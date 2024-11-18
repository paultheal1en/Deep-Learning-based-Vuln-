static void ohci_sysbus_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = ohci_realize_pxa;

    set_bit(DEVICE_CATEGORY_USB, dc->categories);

    dc->desc = "OHCI USB Controller";

    dc->props = ohci_sysbus_properties;

    dc->reset = usb_ohci_reset_sysbus;






}
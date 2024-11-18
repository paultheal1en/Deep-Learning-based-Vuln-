static void isabus_fdc_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = isabus_fdc_realize;

    dc->fw_name = "fdc";

    dc->no_user = 1;

    dc->reset = fdctrl_external_reset_isa;

    dc->vmsd = &vmstate_isa_fdc;

    dc->props = isa_fdc_properties;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);

}

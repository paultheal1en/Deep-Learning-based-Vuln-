static void pcspk_class_initfn(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = pcspk_realizefn;

    set_bit(DEVICE_CATEGORY_SOUND, dc->categories);

    dc->no_user = 1;

    dc->props = pcspk_properties;

}

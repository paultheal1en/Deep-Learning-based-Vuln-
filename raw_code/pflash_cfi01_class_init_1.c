static void pflash_cfi01_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = pflash_cfi01_realize;

    dc->props = pflash_cfi01_properties;

    dc->vmsd = &vmstate_pflash;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);






}
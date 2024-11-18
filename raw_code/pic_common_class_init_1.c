static void pic_common_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->vmsd = &vmstate_pic_common;

    dc->no_user = 1;

    dc->props = pic_properties_common;

    dc->realize = pic_common_realize;

}

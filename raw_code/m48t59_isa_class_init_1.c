static void m48t59_isa_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = m48t59_isa_realize;

    dc->no_user = 1;

    dc->reset = m48t59_reset_isa;

    dc->props = m48t59_isa_properties;

}

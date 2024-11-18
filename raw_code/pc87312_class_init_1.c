static void pc87312_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = pc87312_realize;

    dc->reset = pc87312_reset;

    dc->vmsd = &vmstate_pc87312;

    dc->props = pc87312_properties;



}
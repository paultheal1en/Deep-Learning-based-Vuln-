static void i8042_class_initfn(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = i8042_realizefn;

    dc->no_user = 1;

    dc->vmsd = &vmstate_kbd_isa;

}

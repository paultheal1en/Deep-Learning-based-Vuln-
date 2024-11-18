static void pl080_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);



    dc->no_user = 1;

    dc->vmsd = &vmstate_pl080;

}

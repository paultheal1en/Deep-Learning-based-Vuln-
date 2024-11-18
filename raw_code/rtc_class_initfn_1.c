static void rtc_class_initfn(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = rtc_realizefn;

    dc->no_user = 1;

    dc->vmsd = &vmstate_rtc;

    dc->props = mc146818rtc_properties;

}

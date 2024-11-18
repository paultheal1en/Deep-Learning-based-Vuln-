static void adb_mouse_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    ADBDeviceClass *adc = ADB_DEVICE_CLASS(oc);

    ADBMouseClass *amc = ADB_MOUSE_CLASS(oc);



    amc->parent_realize = dc->realize;

    dc->realize = adb_mouse_realizefn;

    set_bit(DEVICE_CATEGORY_INPUT, dc->categories);



    adc->devreq = adb_mouse_request;

    dc->reset = adb_mouse_reset;

    dc->vmsd = &vmstate_adb_mouse;

}

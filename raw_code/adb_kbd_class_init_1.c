static void adb_kbd_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    ADBDeviceClass *adc = ADB_DEVICE_CLASS(oc);

    ADBKeyboardClass *akc = ADB_KEYBOARD_CLASS(oc);



    akc->parent_realize = dc->realize;

    dc->realize = adb_kbd_realizefn;

    set_bit(DEVICE_CATEGORY_INPUT, dc->categories);



    adc->devreq = adb_kbd_request;

    dc->reset = adb_kbd_reset;

    dc->vmsd = &vmstate_adb_kbd;

}

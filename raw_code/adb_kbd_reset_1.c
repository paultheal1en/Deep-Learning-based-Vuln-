static void adb_kbd_reset(DeviceState *dev)

{

    ADBDevice *d = ADB_DEVICE(dev);

    KBDState *s = ADB_KEYBOARD(dev);



    d->handler = 1;

    d->devaddr = ADB_DEVID_KEYBOARD;

    memset(s->data, 0, sizeof(s->data));

    s->rptr = 0;

    s->wptr = 0;

    s->count = 0;

}

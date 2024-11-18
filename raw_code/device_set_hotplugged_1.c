static void device_set_hotplugged(Object *obj, bool value, Error **err)

{

    DeviceState *dev = DEVICE(obj);



    dev->hotplugged = value;

}

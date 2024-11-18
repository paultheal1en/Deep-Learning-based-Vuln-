static void release_drive(Object *obj, const char *name, void *opaque)

{

    DeviceState *dev = DEVICE(obj);

    Property *prop = opaque;

    BlockDriverState **ptr = qdev_get_prop_ptr(dev, prop);



    if (*ptr) {

        bdrv_detach_dev(*ptr, dev);

        blockdev_auto_del(*ptr);

    }

}

static void free_drive(DeviceState *dev, Property *prop)

{

    DriveInfo **ptr = qdev_get_prop_ptr(dev, prop);



    if (*ptr) {

        blockdev_auto_del((*ptr)->bdrv);

    }

}

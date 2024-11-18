static int parse_drive(DeviceState *dev, Property *prop, const char *str)

{

    DriveInfo **ptr = qdev_get_prop_ptr(dev, prop);



    *ptr = drive_get_by_id(str);

    if (*ptr == NULL)

        return -ENOENT;

    return 0;

}

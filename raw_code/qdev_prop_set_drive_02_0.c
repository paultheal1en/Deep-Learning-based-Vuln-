int qdev_prop_set_drive(DeviceState *dev, const char *name,

                        BlockDriverState *value)

{

    Error *err = NULL;

    const char *bdrv_name = value ? bdrv_get_device_name(value) : "";

    object_property_set_str(OBJECT(dev), bdrv_name,

                            name, &err);

    if (err) {

        qerror_report_err(err);

        error_free(err);

        return -1;

    }

    return 0;

}

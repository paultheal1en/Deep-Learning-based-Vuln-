static char *isabus_get_fw_dev_path(DeviceState *dev)

{

    ISADevice *d = (ISADevice*)dev;

    char path[40];

    int off;



    off = snprintf(path, sizeof(path), "%s", qdev_fw_name(dev));

    if (d->ioport_id) {

        snprintf(path + off, sizeof(path) - off, "@%04x", d->ioport_id);

    }



    return strdup(path);

}

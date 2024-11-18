static char *usb_get_fw_dev_path(DeviceState *qdev)

{

    USBDevice *dev = DO_UPCAST(USBDevice, qdev, qdev);

    char *fw_path, *in;

    int pos = 0;

    long nr;



    fw_path = qemu_malloc(32 + strlen(dev->port->path) * 6);

    in = dev->port->path;

    while (true) {

        nr = strtol(in, &in, 10);

        if (in[0] == '.') {

            /* some hub between root port and device */

            pos += sprintf(fw_path + pos, "hub@%ld/", nr);

            in++;

        } else {

            /* the device itself */

            pos += sprintf(fw_path + pos, "%s@%ld", qdev_fw_name(qdev), nr);

            break;

        }

    }

    return fw_path;

}

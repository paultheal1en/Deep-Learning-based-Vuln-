static USBDevice *usb_net_init(USBBus *bus, const char *cmdline)

{

    Error *local_err = NULL;

    USBDevice *dev;

    QemuOpts *opts;

    int idx;



    opts = qemu_opts_parse(qemu_find_opts("net"), cmdline, 0);

    if (!opts) {

        return NULL;

    }

    qemu_opt_set(opts, "type", "nic");

    qemu_opt_set(opts, "model", "usb");



    idx = net_client_init(opts, 0, &local_err);

    if (local_err) {

        error_report("%s", error_get_pretty(local_err));

        error_free(local_err);

        return NULL;

    }



    dev = usb_create(bus, "usb-net");

    qdev_set_nic_properties(&dev->qdev, &nd_table[idx]);

    qdev_init_nofail(&dev->qdev);

    return dev;

}

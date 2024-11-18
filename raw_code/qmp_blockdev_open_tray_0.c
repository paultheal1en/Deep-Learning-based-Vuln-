void qmp_blockdev_open_tray(const char *device, bool has_force, bool force,

                            Error **errp)

{

    if (!has_force) {

        force = false;

    }

    do_open_tray(device, force, errp);

}

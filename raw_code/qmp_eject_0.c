void qmp_eject(const char *device, bool has_force, bool force, Error **errp)

{

    Error *local_err = NULL;

    int rc;



    if (!has_force) {

        force = false;

    }



    rc = do_open_tray(device, force, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    if (rc == EINPROGRESS) {

        error_setg(errp, "Device '%s' is locked and force was not specified, "

                   "wait for tray to open and try again", device);

        return;

    }



    qmp_x_blockdev_remove_medium(device, errp);

}

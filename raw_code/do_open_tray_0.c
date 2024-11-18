static int do_open_tray(const char *device, bool force, Error **errp)

{

    BlockBackend *blk;

    bool locked;



    blk = blk_by_name(device);

    if (!blk) {

        error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND,

                  "Device '%s' not found", device);

        return -ENODEV;

    }



    if (!blk_dev_has_removable_media(blk)) {

        error_setg(errp, "Device '%s' is not removable", device);

        return -ENOTSUP;

    }



    if (!blk_dev_has_tray(blk)) {

        /* Ignore this command on tray-less devices */

        return ENOSYS;

    }



    if (blk_dev_is_tray_open(blk)) {

        return 0;

    }



    locked = blk_dev_is_medium_locked(blk);

    if (locked) {

        blk_dev_eject_request(blk, force);

    }



    if (!locked || force) {

        blk_dev_change_media_cb(blk, false);

    }



    if (locked && !force) {

        return EINPROGRESS;

    }



    return 0;

}

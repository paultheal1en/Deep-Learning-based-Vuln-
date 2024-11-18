static void eject_device(BlockDriverState *bs, int force, Error **errp)

{

    if (bdrv_in_use(bs)) {

        error_set(errp, QERR_DEVICE_IN_USE, bdrv_get_device_name(bs));

        return;

    }

    if (!bdrv_dev_has_removable_media(bs)) {

        error_setg(errp, "Device '%s' is not removable",

                   bdrv_get_device_name(bs));

        return;

    }



    if (bdrv_dev_is_medium_locked(bs) && !bdrv_dev_is_tray_open(bs)) {

        bdrv_dev_eject_request(bs, force);

        if (!force) {

            error_setg(errp, "Device '%s' is locked",

                       bdrv_get_device_name(bs));

            return;

        }

    }



    bdrv_close(bs);

}

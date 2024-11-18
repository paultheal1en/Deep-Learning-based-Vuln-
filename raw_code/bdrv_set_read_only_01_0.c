int bdrv_set_read_only(BlockDriverState *bs, bool read_only, Error **errp)

{

    /* Do not set read_only if copy_on_read is enabled */

    if (bs->copy_on_read && read_only) {

        error_setg(errp, "Can't set node '%s' to r/o with copy-on-read enabled",

                   bdrv_get_device_or_node_name(bs));

        return -EINVAL;

    }



    /* Do not clear read_only if it is prohibited */

    if (!read_only && !(bs->open_flags & BDRV_O_ALLOW_RDWR)) {

        error_setg(errp, "Node '%s' is read only",

                   bdrv_get_device_or_node_name(bs));

        return -EPERM;

    }



    bs->read_only = read_only;

    return 0;

}

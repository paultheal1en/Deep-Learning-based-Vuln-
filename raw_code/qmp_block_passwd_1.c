void qmp_block_passwd(bool has_device, const char *device,

                      bool has_node_name, const char *node_name,

                      const char *password, Error **errp)

{

    Error *local_err = NULL;

    BlockDriverState *bs;

    int err;



    bs = bdrv_lookup_bs(has_device ? device : NULL,

                        has_node_name ? node_name : NULL,

                        &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    err = bdrv_set_key(bs, password);

    if (err == -EINVAL) {

        error_set(errp, QERR_DEVICE_NOT_ENCRYPTED, bdrv_get_device_name(bs));

        return;

    } else if (err < 0) {

        error_set(errp, QERR_INVALID_PASSWORD);

        return;

    }

}

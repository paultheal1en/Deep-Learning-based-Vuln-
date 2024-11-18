int bdrv_snapshot_load_tmp_by_id_or_name(BlockDriverState *bs,

                                         const char *id_or_name,

                                         Error **errp)

{

    int ret;

    Error *local_err = NULL;



    ret = bdrv_snapshot_load_tmp(bs, id_or_name, NULL, &local_err);

    if (ret == -ENOENT || ret == -EINVAL) {

        error_free(local_err);

        local_err = NULL;

        ret = bdrv_snapshot_load_tmp(bs, NULL, id_or_name, &local_err);

    }



    if (local_err) {

        error_propagate(errp, local_err);

    }



    return ret;

}

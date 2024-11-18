static void qcow2_invalidate_cache(BlockDriverState *bs, Error **errp)

{

    BDRVQcow2State *s = bs->opaque;

    int flags = s->flags;

    QCryptoCipher *cipher = NULL;

    QDict *options;

    Error *local_err = NULL;

    int ret;



    /*

     * Backing files are read-only which makes all of their metadata immutable,

     * that means we don't have to worry about reopening them here.

     */



    cipher = s->cipher;

    s->cipher = NULL;



    qcow2_close(bs);



    bdrv_invalidate_cache(bs->file->bs, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        bs->drv = NULL;

        return;

    }



    memset(s, 0, sizeof(BDRVQcow2State));

    options = qdict_clone_shallow(bs->options);



    flags &= ~BDRV_O_INACTIVE;

    ret = qcow2_open(bs, options, flags, &local_err);

    QDECREF(options);

    if (local_err) {

        error_propagate(errp, local_err);

        error_prepend(errp, "Could not reopen qcow2 layer: ");

        bs->drv = NULL;

        return;

    } else if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not reopen qcow2 layer");

        bs->drv = NULL;

        return;

    }



    s->cipher = cipher;

}

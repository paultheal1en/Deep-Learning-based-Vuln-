static BlockJob *find_block_job(const char *device, AioContext **aio_context,

                                Error **errp)

{

    BlockBackend *blk;

    BlockDriverState *bs;



    *aio_context = NULL;



    blk = blk_by_name(device);

    if (!blk) {

        goto notfound;

    }



    *aio_context = blk_get_aio_context(blk);

    aio_context_acquire(*aio_context);



    if (!blk_is_available(blk)) {

        goto notfound;

    }

    bs = blk_bs(blk);



    if (!bs->job) {

        goto notfound;

    }



    return bs->job;



notfound:

    error_set(errp, ERROR_CLASS_DEVICE_NOT_ACTIVE,

              "No active block job on device '%s'", device);

    if (*aio_context) {

        aio_context_release(*aio_context);

        *aio_context = NULL;

    }

    return NULL;

}

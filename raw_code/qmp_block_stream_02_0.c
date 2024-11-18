void qmp_block_stream(bool has_job_id, const char *job_id, const char *device,

                      bool has_base, const char *base,

                      bool has_backing_file, const char *backing_file,

                      bool has_speed, int64_t speed,

                      bool has_on_error, BlockdevOnError on_error,

                      Error **errp)

{

    BlockDriverState *bs;

    BlockDriverState *base_bs = NULL;

    AioContext *aio_context;

    Error *local_err = NULL;

    const char *base_name = NULL;



    if (!has_on_error) {

        on_error = BLOCKDEV_ON_ERROR_REPORT;

    }



    bs = qmp_get_root_bs(device, errp);

    if (!bs) {

        return;

    }



    aio_context = bdrv_get_aio_context(bs);

    aio_context_acquire(aio_context);



    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_STREAM, errp)) {

        goto out;

    }



    if (has_base) {

        base_bs = bdrv_find_backing_image(bs, base);

        if (base_bs == NULL) {

            error_setg(errp, QERR_BASE_NOT_FOUND, base);

            goto out;

        }

        assert(bdrv_get_aio_context(base_bs) == aio_context);

        base_name = base;

    }



    /* if we are streaming the entire chain, the result will have no backing

     * file, and specifying one is therefore an error */

    if (base_bs == NULL && has_backing_file) {

        error_setg(errp, "backing file specified, but streaming the "

                         "entire chain");

        goto out;

    }



    /* backing_file string overrides base bs filename */

    base_name = has_backing_file ? backing_file : base_name;



    stream_start(has_job_id ? job_id : NULL, bs, base_bs, base_name,

                 has_speed ? speed : 0, on_error, block_job_cb, bs, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto out;

    }



    trace_qmp_block_stream(bs, bs->job);



out:

    aio_context_release(aio_context);

}

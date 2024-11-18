void *block_job_create(const BlockJobDriver *driver, BlockDriverState *bs,

                       int64_t speed, BlockDriverCompletionFunc *cb,

                       void *opaque, Error **errp)

{

    BlockJob *job;



    if (bs->job || bdrv_in_use(bs)) {

        error_set(errp, QERR_DEVICE_IN_USE, bdrv_get_device_name(bs));

        return NULL;

    }

    bdrv_ref(bs);

    bdrv_set_in_use(bs, 1);



    job = g_malloc0(driver->instance_size);

    job->driver        = driver;

    job->bs            = bs;

    job->cb            = cb;

    job->opaque        = opaque;

    job->busy          = true;

    bs->job = job;



    /* Only set speed when necessary to avoid NotSupported error */

    if (speed != 0) {

        Error *local_err = NULL;



        block_job_set_speed(job, speed, &local_err);

        if (local_err) {

            bs->job = NULL;

            g_free(job);

            bdrv_set_in_use(bs, 0);

            error_propagate(errp, local_err);

            return NULL;

        }

    }

    return job;

}

void *block_job_create(const BlockJobDriver *driver, BlockDriverState *bs,

                       int64_t speed, BlockCompletionFunc *cb,

                       void *opaque, Error **errp)

{

    BlockBackend *blk;

    BlockJob *job;



    assert(cb);

    if (bs->job) {

        error_setg(errp, QERR_DEVICE_IN_USE, bdrv_get_device_name(bs));

        return NULL;

    }



    blk = blk_new();

    blk_insert_bs(blk, bs);



    job = g_malloc0(driver->instance_size);

    error_setg(&job->blocker, "block device is in use by block job: %s",

               BlockJobType_lookup[driver->job_type]);

    bdrv_op_block_all(bs, job->blocker);

    bdrv_op_unblock(bs, BLOCK_OP_TYPE_DATAPLANE, job->blocker);



    job->driver        = driver;

    job->id            = g_strdup(bdrv_get_device_name(bs));

    job->blk           = blk;

    job->cb            = cb;

    job->opaque        = opaque;

    job->busy          = true;

    job->refcnt        = 1;

    bs->job = job;



    QLIST_INSERT_HEAD(&block_jobs, job, job_list);



    blk_add_aio_context_notifier(blk, block_job_attached_aio_context,

                                 block_job_detach_aio_context, job);



    /* Only set speed when necessary to avoid NotSupported error */

    if (speed != 0) {

        Error *local_err = NULL;



        block_job_set_speed(job, speed, &local_err);

        if (local_err) {

            block_job_unref(job);

            error_propagate(errp, local_err);

            return NULL;

        }

    }

    return job;

}

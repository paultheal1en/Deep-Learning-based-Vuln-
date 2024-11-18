void *block_job_create(const BlockJobType *job_type, BlockDriverState *bs,

                       BlockDriverCompletionFunc *cb, void *opaque,

                       Error **errp)

{

    BlockJob *job;



    if (bs->job || bdrv_in_use(bs)) {

        error_set(errp, QERR_DEVICE_IN_USE, bdrv_get_device_name(bs));

        return NULL;

    }

    bdrv_set_in_use(bs, 1);



    job = g_malloc0(job_type->instance_size);

    job->job_type      = job_type;

    job->bs            = bs;

    job->cb            = cb;

    job->opaque        = opaque;

    bs->job = job;

    return job;

}

void block_job_completed(BlockJob *job, int ret)

{

    BlockDriverState *bs = job->bs;



    assert(bs->job == job);

    job->cb(job->opaque, ret);

    bs->job = NULL;

    bdrv_op_unblock_all(bs, job->blocker);

    error_free(job->blocker);

    g_free(job);

}

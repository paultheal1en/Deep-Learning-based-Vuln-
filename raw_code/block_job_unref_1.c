static void block_job_unref(BlockJob *job)

{

    if (--job->refcnt == 0) {

        BlockDriverState *bs = blk_bs(job->blk);

        bs->job = NULL;

        block_job_remove_all_bdrv(job);

        blk_remove_aio_context_notifier(job->blk,

                                        block_job_attached_aio_context,

                                        block_job_detach_aio_context, job);

        blk_unref(job->blk);

        error_free(job->blocker);

        g_free(job->id);

        QLIST_REMOVE(job, job_list);

        g_free(job);

    }

}

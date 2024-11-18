static void commit_complete(BlockJob *job, void *opaque)

{

    CommitBlockJob *s = container_of(job, CommitBlockJob, common);

    CommitCompleteData *data = opaque;

    BlockDriverState *active = s->active;

    BlockDriverState *top = blk_bs(s->top);

    BlockDriverState *base = blk_bs(s->base);

    BlockDriverState *overlay_bs = bdrv_find_overlay(active, top);

    int ret = data->ret;



    if (!block_job_is_cancelled(&s->common) && ret == 0) {

        /* success */

        ret = bdrv_drop_intermediate(active, top, base, s->backing_file_str);

    }



    /* restore base open flags here if appropriate (e.g., change the base back

     * to r/o). These reopens do not need to be atomic, since we won't abort

     * even on failure here */

    if (s->base_flags != bdrv_get_flags(base)) {

        bdrv_reopen(base, s->base_flags, NULL);

    }

    if (overlay_bs && s->orig_overlay_flags != bdrv_get_flags(overlay_bs)) {

        bdrv_reopen(overlay_bs, s->orig_overlay_flags, NULL);

    }

    g_free(s->backing_file_str);

    blk_unref(s->top);

    blk_unref(s->base);

    block_job_completed(&s->common, ret);

    g_free(data);

}

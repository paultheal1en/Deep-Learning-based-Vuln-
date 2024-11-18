static void stream_complete(BlockJob *job, void *opaque)

{

    StreamBlockJob *s = container_of(job, StreamBlockJob, common);

    StreamCompleteData *data = opaque;

    BlockDriverState *bs = blk_bs(job->blk);

    BlockDriverState *base = s->base;

    Error *local_err = NULL;



    if (!block_job_is_cancelled(&s->common) && data->reached_end &&

        data->ret == 0) {

        const char *base_id = NULL, *base_fmt = NULL;

        if (base) {

            base_id = s->backing_file_str;

            if (base->drv) {

                base_fmt = base->drv->format_name;

            }

        }

        data->ret = bdrv_change_backing_file(bs, base_id, base_fmt);

        bdrv_set_backing_hd(bs, base, &local_err);

        if (local_err) {

            error_report_err(local_err);

            data->ret = -EPERM;

            goto out;

        }

    }



out:

    /* Reopen the image back in read-only mode if necessary */

    if (s->bs_flags != bdrv_get_flags(bs)) {

        /* Give up write permissions before making it read-only */

        blk_set_perm(job->blk, 0, BLK_PERM_ALL, &error_abort);

        bdrv_reopen(bs, s->bs_flags, NULL);

    }



    g_free(s->backing_file_str);

    block_job_completed(&s->common, data->ret);

    g_free(data);

}

static void mirror_exit(BlockJob *job, void *opaque)

{

    MirrorBlockJob *s = container_of(job, MirrorBlockJob, common);

    MirrorExitData *data = opaque;

    AioContext *replace_aio_context = NULL;

    BlockDriverState *src = s->common.bs;



    /* Make sure that the source BDS doesn't go away before we called

     * block_job_completed(). */

    bdrv_ref(src);



    if (s->to_replace) {

        replace_aio_context = bdrv_get_aio_context(s->to_replace);

        aio_context_acquire(replace_aio_context);

    }



    if (s->should_complete && data->ret == 0) {

        BlockDriverState *to_replace = s->common.bs;

        if (s->to_replace) {

            to_replace = s->to_replace;

        }



        /* This was checked in mirror_start_job(), but meanwhile one of the

         * nodes could have been newly attached to a BlockBackend. */

        if (to_replace->blk && s->target->blk) {

            error_report("block job: Can't create node with two BlockBackends");

            data->ret = -EINVAL;

            goto out;

        }



        if (bdrv_get_flags(s->target) != bdrv_get_flags(to_replace)) {

            bdrv_reopen(s->target, bdrv_get_flags(to_replace), NULL);

        }

        bdrv_replace_in_backing_chain(to_replace, s->target);

    }



out:

    if (s->to_replace) {

        bdrv_op_unblock_all(s->to_replace, s->replace_blocker);

        error_free(s->replace_blocker);

        bdrv_unref(s->to_replace);

    }

    if (replace_aio_context) {

        aio_context_release(replace_aio_context);

    }

    g_free(s->replaces);

    bdrv_op_unblock_all(s->target, s->common.blocker);

    bdrv_unref(s->target);

    block_job_completed(&s->common, data->ret);

    g_free(data);

    bdrv_drained_end(src);

    if (qemu_get_aio_context() == bdrv_get_aio_context(src)) {

        aio_enable_external(iohandler_get_aio_context());

    }

    bdrv_unref(src);

}

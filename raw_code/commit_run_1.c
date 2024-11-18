static void coroutine_fn commit_run(void *opaque)

{

    CommitBlockJob *s = opaque;

    BlockDriverState *active = s->active;

    BlockDriverState *top = s->top;

    BlockDriverState *base = s->base;

    BlockDriverState *overlay_bs = NULL;

    int64_t sector_num, end;

    int ret = 0;

    int n = 0;

    void *buf;

    int bytes_written = 0;

    int64_t base_len;



    ret = s->common.len = bdrv_getlength(top);





    if (s->common.len < 0) {

        goto exit_restore_reopen;

    }



    ret = base_len = bdrv_getlength(base);

    if (base_len < 0) {

        goto exit_restore_reopen;

    }



    if (base_len < s->common.len) {

        ret = bdrv_truncate(base, s->common.len);

        if (ret) {

            goto exit_restore_reopen;

        }

    }



    overlay_bs = bdrv_find_overlay(active, top);



    end = s->common.len >> BDRV_SECTOR_BITS;

    buf = qemu_blockalign(top, COMMIT_BUFFER_SIZE);



    for (sector_num = 0; sector_num < end; sector_num += n) {

        uint64_t delay_ns = 0;

        bool copy;



wait:

        /* Note that even when no rate limit is applied we need to yield

         * with no pending I/O here so that bdrv_drain_all() returns.

         */

        block_job_sleep_ns(&s->common, rt_clock, delay_ns);

        if (block_job_is_cancelled(&s->common)) {

            break;

        }

        /* Copy if allocated above the base */

        ret = bdrv_co_is_allocated_above(top, base, sector_num,

                                         COMMIT_BUFFER_SIZE / BDRV_SECTOR_SIZE,

                                         &n);

        copy = (ret == 1);

        trace_commit_one_iteration(s, sector_num, n, ret);

        if (copy) {

            if (s->common.speed) {

                delay_ns = ratelimit_calculate_delay(&s->limit, n);

                if (delay_ns > 0) {

                    goto wait;

                }

            }

            ret = commit_populate(top, base, sector_num, n, buf);

            bytes_written += n * BDRV_SECTOR_SIZE;

        }

        if (ret < 0) {

            if (s->on_error == BLOCKDEV_ON_ERROR_STOP ||

                s->on_error == BLOCKDEV_ON_ERROR_REPORT||

                (s->on_error == BLOCKDEV_ON_ERROR_ENOSPC && ret == -ENOSPC)) {

                goto exit_free_buf;

            } else {

                n = 0;

                continue;

            }

        }

        /* Publish progress */

        s->common.offset += n * BDRV_SECTOR_SIZE;

    }



    ret = 0;



    if (!block_job_is_cancelled(&s->common) && sector_num == end) {

        /* success */

        ret = bdrv_drop_intermediate(active, top, base);

    }



exit_free_buf:

    qemu_vfree(buf);



exit_restore_reopen:

    /* restore base open flags here if appropriate (e.g., change the base back

     * to r/o). These reopens do not need to be atomic, since we won't abort

     * even on failure here */

    if (s->base_flags != bdrv_get_flags(base)) {

        bdrv_reopen(base, s->base_flags, NULL);

    }

    if (s->orig_overlay_flags != bdrv_get_flags(overlay_bs)) {

        bdrv_reopen(overlay_bs, s->orig_overlay_flags, NULL);

    }



    block_job_completed(&s->common, ret);

}

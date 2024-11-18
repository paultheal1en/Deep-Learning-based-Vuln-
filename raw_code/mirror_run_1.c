static void coroutine_fn mirror_run(void *opaque)

{

    MirrorBlockJob *s = opaque;

    MirrorExitData *data;

    BlockDriverState *bs = s->common.bs;

    int64_t sector_num, end, sectors_per_chunk, length;

    uint64_t last_pause_ns;

    BlockDriverInfo bdi;

    char backing_filename[1024];

    int ret = 0;

    int n;



    if (block_job_is_cancelled(&s->common)) {

        goto immediate_exit;

    }



    s->bdev_length = bdrv_getlength(bs);

    if (s->bdev_length < 0) {

        ret = s->bdev_length;

        goto immediate_exit;

    } else if (s->bdev_length == 0) {

        /* Report BLOCK_JOB_READY and wait for complete. */

        block_job_event_ready(&s->common);

        s->synced = true;

        while (!block_job_is_cancelled(&s->common) && !s->should_complete) {

            block_job_yield(&s->common);

        }

        s->common.cancelled = false;

        goto immediate_exit;

    }



    length = DIV_ROUND_UP(s->bdev_length, s->granularity);

    s->in_flight_bitmap = bitmap_new(length);



    /* If we have no backing file yet in the destination, we cannot let

     * the destination do COW.  Instead, we copy sectors around the

     * dirty data if needed.  We need a bitmap to do that.

     */

    bdrv_get_backing_filename(s->target, backing_filename,

                              sizeof(backing_filename));

    if (backing_filename[0] && !s->target->backing_hd) {

        ret = bdrv_get_info(s->target, &bdi);

        if (ret < 0) {

            goto immediate_exit;

        }

        if (s->granularity < bdi.cluster_size) {

            s->buf_size = MAX(s->buf_size, bdi.cluster_size);

            s->cow_bitmap = bitmap_new(length);

        }

    }



    end = s->bdev_length / BDRV_SECTOR_SIZE;

    s->buf = qemu_try_blockalign(bs, s->buf_size);

    if (s->buf == NULL) {

        ret = -ENOMEM;

        goto immediate_exit;

    }



    sectors_per_chunk = s->granularity >> BDRV_SECTOR_BITS;

    mirror_free_init(s);



    if (!s->is_none_mode) {

        /* First part, loop on the sectors and initialize the dirty bitmap.  */

        BlockDriverState *base = s->base;

        for (sector_num = 0; sector_num < end; ) {

            int64_t next = (sector_num | (sectors_per_chunk - 1)) + 1;

            ret = bdrv_is_allocated_above(bs, base,

                                          sector_num, next - sector_num, &n);



            if (ret < 0) {

                goto immediate_exit;

            }



            assert(n > 0);

            if (ret == 1) {

                bdrv_set_dirty(bs, sector_num, n);

                sector_num = next;

            } else {

                sector_num += n;

            }

        }

    }



    bdrv_dirty_iter_init(bs, s->dirty_bitmap, &s->hbi);

    last_pause_ns = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);

    for (;;) {

        uint64_t delay_ns = 0;

        int64_t cnt;

        bool should_complete;



        if (s->ret < 0) {

            ret = s->ret;

            goto immediate_exit;

        }



        cnt = bdrv_get_dirty_count(bs, s->dirty_bitmap);

        /* s->common.offset contains the number of bytes already processed so

         * far, cnt is the number of dirty sectors remaining and

         * s->sectors_in_flight is the number of sectors currently being

         * processed; together those are the current total operation length */

        s->common.len = s->common.offset +

                        (cnt + s->sectors_in_flight) * BDRV_SECTOR_SIZE;



        /* Note that even when no rate limit is applied we need to yield

         * periodically with no pending I/O so that qemu_aio_flush() returns.

         * We do so every SLICE_TIME nanoseconds, or when there is an error,

         * or when the source is clean, whichever comes first.

         */

        if (qemu_clock_get_ns(QEMU_CLOCK_REALTIME) - last_pause_ns < SLICE_TIME &&

            s->common.iostatus == BLOCK_DEVICE_IO_STATUS_OK) {

            if (s->in_flight == MAX_IN_FLIGHT || s->buf_free_count == 0 ||

                (cnt == 0 && s->in_flight > 0)) {

                trace_mirror_yield(s, s->in_flight, s->buf_free_count, cnt);

                qemu_coroutine_yield();

                continue;

            } else if (cnt != 0) {

                delay_ns = mirror_iteration(s);

                if (delay_ns == 0) {

                    continue;

                }

            }

        }



        should_complete = false;

        if (s->in_flight == 0 && cnt == 0) {

            trace_mirror_before_flush(s);

            ret = bdrv_flush(s->target);

            if (ret < 0) {

                if (mirror_error_action(s, false, -ret) ==

                    BLOCK_ERROR_ACTION_REPORT) {

                    goto immediate_exit;

                }

            } else {

                /* We're out of the streaming phase.  From now on, if the job

                 * is cancelled we will actually complete all pending I/O and

                 * report completion.  This way, block-job-cancel will leave

                 * the target in a consistent state.

                 */

                if (!s->synced) {

                    block_job_event_ready(&s->common);

                    s->synced = true;

                }



                should_complete = s->should_complete ||

                    block_job_is_cancelled(&s->common);

                cnt = bdrv_get_dirty_count(bs, s->dirty_bitmap);

            }

        }



        if (cnt == 0 && should_complete) {

            /* The dirty bitmap is not updated while operations are pending.

             * If we're about to exit, wait for pending operations before

             * calling bdrv_get_dirty_count(bs), or we may exit while the

             * source has dirty data to copy!

             *

             * Note that I/O can be submitted by the guest while

             * mirror_populate runs.

             */

            trace_mirror_before_drain(s, cnt);

            bdrv_drain(bs);

            cnt = bdrv_get_dirty_count(bs, s->dirty_bitmap);

        }



        ret = 0;

        trace_mirror_before_sleep(s, cnt, s->synced, delay_ns);

        if (!s->synced) {

            block_job_sleep_ns(&s->common, QEMU_CLOCK_REALTIME, delay_ns);

            if (block_job_is_cancelled(&s->common)) {

                break;

            }

        } else if (!should_complete) {

            delay_ns = (s->in_flight == 0 && cnt == 0 ? SLICE_TIME : 0);

            block_job_sleep_ns(&s->common, QEMU_CLOCK_REALTIME, delay_ns);

        } else if (cnt == 0) {

            /* The two disks are in sync.  Exit and report successful

             * completion.

             */

            assert(QLIST_EMPTY(&bs->tracked_requests));

            s->common.cancelled = false;

            break;

        }

        last_pause_ns = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);

    }



immediate_exit:

    if (s->in_flight > 0) {

        /* We get here only if something went wrong.  Either the job failed,

         * or it was cancelled prematurely so that we do not guarantee that

         * the target is a copy of the source.

         */

        assert(ret < 0 || (!s->synced && block_job_is_cancelled(&s->common)));

        mirror_drain(s);

    }



    assert(s->in_flight == 0);

    qemu_vfree(s->buf);

    g_free(s->cow_bitmap);

    g_free(s->in_flight_bitmap);

    bdrv_release_dirty_bitmap(bs, s->dirty_bitmap);

    bdrv_iostatus_disable(s->target);



    data = g_malloc(sizeof(*data));

    data->ret = ret;

    block_job_defer_to_main_loop(&s->common, mirror_exit, data);

}

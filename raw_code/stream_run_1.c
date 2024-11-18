static void coroutine_fn stream_run(void *opaque)

{

    StreamBlockJob *s = opaque;

    BlockDriverState *bs = s->common.bs;

    BlockDriverState *base = s->base;

    int64_t sector_num, end;

    int error = 0;

    int ret = 0;

    int n = 0;

    void *buf;



    s->common.len = bdrv_getlength(bs);

    if (s->common.len < 0) {

        block_job_completed(&s->common, s->common.len);

        return;

    }



    end = s->common.len >> BDRV_SECTOR_BITS;

    buf = qemu_blockalign(bs, STREAM_BUFFER_SIZE);



    /* Turn on copy-on-read for the whole block device so that guest read

     * requests help us make progress.  Only do this when copying the entire

     * backing chain since the copy-on-read operation does not take base into

     * account.

     */

    if (!base) {

        bdrv_enable_copy_on_read(bs);

    }



    for (sector_num = 0; sector_num < end; sector_num += n) {

        uint64_t delay_ns = 0;

        bool copy;



wait:

        /* Note that even when no rate limit is applied we need to yield

         * with no pending I/O here so that bdrv_drain_all() returns.

         */

        block_job_sleep_ns(&s->common, QEMU_CLOCK_REALTIME, delay_ns);

        if (block_job_is_cancelled(&s->common)) {

            break;

        }



        ret = bdrv_is_allocated(bs, sector_num,

                                STREAM_BUFFER_SIZE / BDRV_SECTOR_SIZE, &n);

        if (ret == 1) {

            /* Allocated in the top, no need to copy.  */

            copy = false;

        } else if (ret >= 0) {

            /* Copy if allocated in the intermediate images.  Limit to the

             * known-unallocated area [sector_num, sector_num+n).  */

            ret = bdrv_is_allocated_above(bs->backing_hd, base,

                                          sector_num, n, &n);



            /* Finish early if end of backing file has been reached */

            if (ret == 0 && n == 0) {

                n = end - sector_num;

            }



            copy = (ret == 1);

        }

        trace_stream_one_iteration(s, sector_num, n, ret);

        if (ret >= 0 && copy) {

            if (s->common.speed) {

                delay_ns = ratelimit_calculate_delay(&s->limit, n);

                if (delay_ns > 0) {

                    goto wait;

                }

            }

            ret = stream_populate(bs, sector_num, n, buf);

        }

        if (ret < 0) {

            BlockErrorAction action =

                block_job_error_action(&s->common, s->common.bs, s->on_error,

                                       true, -ret);

            if (action == BDRV_ACTION_STOP) {

                n = 0;

                continue;

            }

            if (error == 0) {

                error = ret;

            }

            if (action == BDRV_ACTION_REPORT) {

                break;

            }

        }

        ret = 0;



        /* Publish progress */

        s->common.offset += n * BDRV_SECTOR_SIZE;

    }



    if (!base) {

        bdrv_disable_copy_on_read(bs);

    }



    /* Do not remove the backing file if an error was there but ignored.  */

    ret = error;



    if (!block_job_is_cancelled(&s->common) && sector_num == end && ret == 0) {

        const char *base_id = NULL, *base_fmt = NULL;

        if (base) {

            base_id = s->backing_file_id;

            if (base->drv) {

                base_fmt = base->drv->format_name;

            }

        }

        ret = bdrv_change_backing_file(bs, base_id, base_fmt);

        close_unused_images(bs, base, base_id);

    }



    qemu_vfree(buf);

    block_job_completed(&s->common, ret);

}

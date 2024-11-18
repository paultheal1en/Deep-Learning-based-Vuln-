static int coroutine_fn backup_do_cow(BackupBlockJob *job,

                                      int64_t offset, uint64_t bytes,

                                      bool *error_is_read,

                                      bool is_write_notifier)

{

    BlockBackend *blk = job->common.blk;

    CowRequest cow_request;

    struct iovec iov;

    QEMUIOVector bounce_qiov;

    void *bounce_buffer = NULL;

    int ret = 0;

    int64_t start, end; /* bytes */

    int n; /* bytes */



    qemu_co_rwlock_rdlock(&job->flush_rwlock);



    start = QEMU_ALIGN_DOWN(offset, job->cluster_size);

    end = QEMU_ALIGN_UP(bytes + offset, job->cluster_size);



    trace_backup_do_cow_enter(job, start, offset, bytes);



    wait_for_overlapping_requests(job, start, end);

    cow_request_begin(&cow_request, job, start, end);



    for (; start < end; start += job->cluster_size) {

        if (test_bit(start / job->cluster_size, job->done_bitmap)) {

            trace_backup_do_cow_skip(job, start);

            continue; /* already copied */

        }



        trace_backup_do_cow_process(job, start);



        n = MIN(job->cluster_size, job->common.len - start);



        if (!bounce_buffer) {

            bounce_buffer = blk_blockalign(blk, job->cluster_size);

        }

        iov.iov_base = bounce_buffer;

        iov.iov_len = n;

        qemu_iovec_init_external(&bounce_qiov, &iov, 1);



        ret = blk_co_preadv(blk, start, bounce_qiov.size, &bounce_qiov,

                            is_write_notifier ? BDRV_REQ_NO_SERIALISING : 0);

        if (ret < 0) {

            trace_backup_do_cow_read_fail(job, start, ret);

            if (error_is_read) {

                *error_is_read = true;

            }

            goto out;

        }



        if (buffer_is_zero(iov.iov_base, iov.iov_len)) {

            ret = blk_co_pwrite_zeroes(job->target, start,

                                       bounce_qiov.size, BDRV_REQ_MAY_UNMAP);

        } else {

            ret = blk_co_pwritev(job->target, start,

                                 bounce_qiov.size, &bounce_qiov,

                                 job->compress ? BDRV_REQ_WRITE_COMPRESSED : 0);

        }

        if (ret < 0) {

            trace_backup_do_cow_write_fail(job, start, ret);

            if (error_is_read) {

                *error_is_read = false;

            }

            goto out;

        }



        set_bit(start / job->cluster_size, job->done_bitmap);



        /* Publish progress, guest I/O counts as progress too.  Note that the

         * offset field is an opaque progress value, it is not a disk offset.

         */

        job->bytes_read += n;

        job->common.offset += n;

    }



out:

    if (bounce_buffer) {

        qemu_vfree(bounce_buffer);

    }



    cow_request_end(&cow_request);



    trace_backup_do_cow_return(job, offset, bytes, ret);



    qemu_co_rwlock_unlock(&job->flush_rwlock);



    return ret;

}

static int coroutine_fn bdrv_co_do_zero_pwritev(BlockDriverState *bs,

                                                int64_t offset,

                                                unsigned int bytes,

                                                BdrvRequestFlags flags,

                                                BdrvTrackedRequest *req)

{

    uint8_t *buf = NULL;

    QEMUIOVector local_qiov;

    struct iovec iov;

    uint64_t align = bs->bl.request_alignment;

    unsigned int head_padding_bytes, tail_padding_bytes;

    int ret = 0;



    head_padding_bytes = offset & (align - 1);

    tail_padding_bytes = align - ((offset + bytes) & (align - 1));





    assert(flags & BDRV_REQ_ZERO_WRITE);

    if (head_padding_bytes || tail_padding_bytes) {

        buf = qemu_blockalign(bs, align);

        iov = (struct iovec) {

            .iov_base   = buf,

            .iov_len    = align,

        };

        qemu_iovec_init_external(&local_qiov, &iov, 1);

    }

    if (head_padding_bytes) {

        uint64_t zero_bytes = MIN(bytes, align - head_padding_bytes);



        /* RMW the unaligned part before head. */

        mark_request_serialising(req, align);

        wait_serialising_requests(req);

        bdrv_debug_event(bs, BLKDBG_PWRITEV_RMW_HEAD);

        ret = bdrv_aligned_preadv(bs, req, offset & ~(align - 1), align,

                                  align, &local_qiov, 0);

        if (ret < 0) {

            goto fail;

        }

        bdrv_debug_event(bs, BLKDBG_PWRITEV_RMW_AFTER_HEAD);



        memset(buf + head_padding_bytes, 0, zero_bytes);

        ret = bdrv_aligned_pwritev(bs, req, offset & ~(align - 1), align,

                                   align, &local_qiov,

                                   flags & ~BDRV_REQ_ZERO_WRITE);

        if (ret < 0) {

            goto fail;

        }

        offset += zero_bytes;

        bytes -= zero_bytes;

    }



    assert(!bytes || (offset & (align - 1)) == 0);

    if (bytes >= align) {

        /* Write the aligned part in the middle. */

        uint64_t aligned_bytes = bytes & ~(align - 1);

        ret = bdrv_aligned_pwritev(bs, req, offset, aligned_bytes, align,

                                   NULL, flags);

        if (ret < 0) {

            goto fail;

        }

        bytes -= aligned_bytes;

        offset += aligned_bytes;

    }



    assert(!bytes || (offset & (align - 1)) == 0);

    if (bytes) {

        assert(align == tail_padding_bytes + bytes);

        /* RMW the unaligned part after tail. */

        mark_request_serialising(req, align);

        wait_serialising_requests(req);

        bdrv_debug_event(bs, BLKDBG_PWRITEV_RMW_TAIL);

        ret = bdrv_aligned_preadv(bs, req, offset, align,

                                  align, &local_qiov, 0);

        if (ret < 0) {

            goto fail;

        }

        bdrv_debug_event(bs, BLKDBG_PWRITEV_RMW_AFTER_TAIL);



        memset(buf, 0, bytes);

        ret = bdrv_aligned_pwritev(bs, req, offset, align, align,

                                   &local_qiov, flags & ~BDRV_REQ_ZERO_WRITE);

    }

fail:

    qemu_vfree(buf);

    return ret;



}

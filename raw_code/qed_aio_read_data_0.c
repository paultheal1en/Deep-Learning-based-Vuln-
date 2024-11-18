static int coroutine_fn qed_aio_read_data(void *opaque, int ret,

                                          uint64_t offset, size_t len)

{

    QEDAIOCB *acb = opaque;

    BDRVQEDState *s = acb_to_s(acb);

    BlockDriverState *bs = acb->bs;



    /* Adjust offset into cluster */

    offset += qed_offset_into_cluster(s, acb->cur_pos);



    trace_qed_aio_read_data(s, acb, ret, offset, len);



    qemu_iovec_concat(&acb->cur_qiov, acb->qiov, acb->qiov_offset, len);



    /* Handle zero cluster and backing file reads */

    if (ret == QED_CLUSTER_ZERO) {

        qemu_iovec_memset(&acb->cur_qiov, 0, 0, acb->cur_qiov.size);

        return 0;

    } else if (ret != QED_CLUSTER_FOUND) {

        return qed_read_backing_file(s, acb->cur_pos, &acb->cur_qiov,

                                     &acb->backing_qiov);

    }



    BLKDBG_EVENT(bs->file, BLKDBG_READ_AIO);

    ret = bdrv_co_preadv(bs->file, offset, acb->cur_qiov.size,

                         &acb->cur_qiov, 0);

    if (ret < 0) {

        return ret;

    }

    return 0;

}

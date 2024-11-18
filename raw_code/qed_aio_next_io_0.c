static void qed_aio_next_io(QEDAIOCB *acb)

{

    BDRVQEDState *s = acb_to_s(acb);

    uint64_t offset;

    size_t len;

    int ret;



    trace_qed_aio_next_io(s, acb, 0, acb->cur_pos + acb->cur_qiov.size);



    if (acb->backing_qiov) {

        qemu_iovec_destroy(acb->backing_qiov);

        g_free(acb->backing_qiov);

        acb->backing_qiov = NULL;

    }



    acb->qiov_offset += acb->cur_qiov.size;

    acb->cur_pos += acb->cur_qiov.size;

    qemu_iovec_reset(&acb->cur_qiov);



    /* Complete request */

    if (acb->cur_pos >= acb->end_pos) {

        qed_aio_complete(acb, 0);

        return;

    }



    /* Find next cluster and start I/O */

    len = acb->end_pos - acb->cur_pos;

    ret = qed_find_cluster(s, &acb->request, acb->cur_pos, &len, &offset);

    if (ret < 0) {

        qed_aio_complete(acb, ret);

        return;

    }



    if (acb->flags & QED_AIOCB_WRITE) {

        ret = qed_aio_write_data(acb, ret, offset, len);

    } else {

        ret = qed_aio_read_data(acb, ret, offset, len);

    }



    if (ret < 0) {

        if (ret != -EINPROGRESS) {

            qed_aio_complete(acb, ret);

        }

        return;

    }

    qed_aio_next_io(acb);

}

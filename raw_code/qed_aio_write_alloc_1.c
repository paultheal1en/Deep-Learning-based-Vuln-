static void qed_aio_write_alloc(QEDAIOCB *acb, size_t len)

{

    BDRVQEDState *s = acb_to_s(acb);



    /* Freeze this request if another allocating write is in progress */

    if (acb != QSIMPLEQ_FIRST(&s->allocating_write_reqs)) {

        QSIMPLEQ_INSERT_TAIL(&s->allocating_write_reqs, acb, next);

    }

    if (acb != QSIMPLEQ_FIRST(&s->allocating_write_reqs)) {

        return; /* wait for existing request to finish */

    }



    acb->cur_nclusters = qed_bytes_to_clusters(s,

            qed_offset_into_cluster(s, acb->cur_pos) + len);

    acb->cur_cluster = qed_alloc_clusters(s, acb->cur_nclusters);

    qemu_iovec_copy(&acb->cur_qiov, acb->qiov, acb->qiov_offset, len);



    if (qed_should_set_need_check(s)) {

        s->header.features |= QED_F_NEED_CHECK;

        qed_write_header(s, qed_aio_write_prefill, acb);

    } else {

        qed_aio_write_prefill(acb, 0);

    }

}

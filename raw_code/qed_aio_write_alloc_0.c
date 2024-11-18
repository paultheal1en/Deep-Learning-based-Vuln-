static int coroutine_fn qed_aio_write_alloc(QEDAIOCB *acb, size_t len)

{

    BDRVQEDState *s = acb_to_s(acb);

    int ret;



    /* Cancel timer when the first allocating request comes in */

    if (s->allocating_acb == NULL) {

        qed_cancel_need_check_timer(s);

    }



    /* Freeze this request if another allocating write is in progress */

    if (s->allocating_acb != acb || s->allocating_write_reqs_plugged) {

        if (s->allocating_acb != NULL) {

            qemu_co_queue_wait(&s->allocating_write_reqs, NULL);

            assert(s->allocating_acb == NULL);

        }

        s->allocating_acb = acb;

        return -EAGAIN; /* start over with looking up table entries */

    }



    acb->cur_nclusters = qed_bytes_to_clusters(s,

            qed_offset_into_cluster(s, acb->cur_pos) + len);

    qemu_iovec_concat(&acb->cur_qiov, acb->qiov, acb->qiov_offset, len);



    if (acb->flags & QED_AIOCB_ZERO) {

        /* Skip ahead if the clusters are already zero */

        if (acb->find_cluster_ret == QED_CLUSTER_ZERO) {

            return 0;

        }

        acb->cur_cluster = 1;

    } else {

        acb->cur_cluster = qed_alloc_clusters(s, acb->cur_nclusters);

    }



    if (qed_should_set_need_check(s)) {

        s->header.features |= QED_F_NEED_CHECK;

        ret = qed_write_header(s);

        if (ret < 0) {

            return ret;

        }

    }



    if (!(acb->flags & QED_AIOCB_ZERO)) {

        ret = qed_aio_write_cow(acb);

        if (ret < 0) {

            return ret;

        }

    }



    return qed_aio_write_l2_update(acb, acb->cur_cluster);

}

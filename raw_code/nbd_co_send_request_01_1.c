static int nbd_co_send_request(BlockDriverState *bs,

                               NBDRequest *request,

                               QEMUIOVector *qiov)

{

    NBDClientSession *s = nbd_get_client_session(bs);

    int rc, ret, i;



    qemu_co_mutex_lock(&s->send_mutex);

    while (s->in_flight == MAX_NBD_REQUESTS) {

        qemu_co_queue_wait(&s->free_sema, &s->send_mutex);

    }

    s->in_flight++;



    for (i = 0; i < MAX_NBD_REQUESTS; i++) {

        if (s->recv_coroutine[i] == NULL) {

            s->recv_coroutine[i] = qemu_coroutine_self();

            break;

        }

    }



    g_assert(qemu_in_coroutine());

    assert(i < MAX_NBD_REQUESTS);

    request->handle = INDEX_TO_HANDLE(s, i);



    if (!s->ioc) {

        qemu_co_mutex_unlock(&s->send_mutex);

        return -EPIPE;

    }



    if (qiov) {

        qio_channel_set_cork(s->ioc, true);

        rc = nbd_send_request(s->ioc, request);

        if (rc >= 0) {

            ret = nbd_rwv(s->ioc, qiov->iov, qiov->niov, request->len, false,

                          NULL);

            if (ret != request->len) {

                rc = -EIO;

            }

        }

        qio_channel_set_cork(s->ioc, false);

    } else {

        rc = nbd_send_request(s->ioc, request);

    }

    qemu_co_mutex_unlock(&s->send_mutex);

    return rc;

}

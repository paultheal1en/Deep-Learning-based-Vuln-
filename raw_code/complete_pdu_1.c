static void complete_pdu(V9fsState *s, V9fsPDU *pdu, ssize_t len)

{

    int8_t id = pdu->id + 1; /* Response */



    if (len < 0) {

        int err = -len;

        len = 7;



        if (s->proto_version != V9FS_PROTO_2000L) {

            V9fsString str;



            str.data = strerror(err);

            str.size = strlen(str.data);



            len += pdu_marshal(pdu, len, "s", &str);

            id = P9_RERROR;

        }



        len += pdu_marshal(pdu, len, "d", err);



        if (s->proto_version == V9FS_PROTO_2000L) {

            id = P9_RLERROR;

        }


    }



    /* fill out the header */

    pdu_marshal(pdu, 0, "dbw", (int32_t)len, id, pdu->tag);



    /* keep these in sync */

    pdu->size = len;

    pdu->id = id;



    /* push onto queue and notify */

    virtqueue_push(s->vq, &pdu->elem, len);



    /* FIXME: we should batch these completions */

    virtio_notify(&s->vdev, s->vq);



    /* Now wakeup anybody waiting in flush for this request */

    qemu_co_queue_next(&pdu->complete);



    free_pdu(s, pdu);

}
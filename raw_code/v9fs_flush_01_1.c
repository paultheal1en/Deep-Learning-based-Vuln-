static void coroutine_fn v9fs_flush(void *opaque)

{

    ssize_t err;

    int16_t tag;

    size_t offset = 7;

    V9fsPDU *cancel_pdu;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    err = pdu_unmarshal(pdu, offset, "w", &tag);

    if (err < 0) {

        pdu_complete(pdu, err);

        return;

    }

    trace_v9fs_flush(pdu->tag, pdu->id, tag);



    QLIST_FOREACH(cancel_pdu, &s->active_list, next) {

        if (cancel_pdu->tag == tag) {

            break;

        }

    }

    if (cancel_pdu) {

        cancel_pdu->cancelled = 1;

        /*

         * Wait for pdu to complete.

         */

        qemu_co_queue_wait(&cancel_pdu->complete, NULL);

        cancel_pdu->cancelled = 0;

        pdu_free(cancel_pdu);

    }

    pdu_complete(pdu, 7);

}

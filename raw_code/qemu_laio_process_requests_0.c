static int qemu_laio_process_requests(void *opaque)

{

    struct qemu_laio_state *s = opaque;

    struct qemu_laiocb *laiocb, *next;

    int res = 0;



    QLIST_FOREACH_SAFE (laiocb, &s->completed_reqs, node, next) {

        if (laiocb->async_context_id == get_async_context_id()) {

            qemu_laio_process_completion(s, laiocb);

            QLIST_REMOVE(laiocb, node);

            res = 1;

        }

    }



    return res;

}

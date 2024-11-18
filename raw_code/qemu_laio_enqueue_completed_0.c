static void qemu_laio_enqueue_completed(struct qemu_laio_state *s,

    struct qemu_laiocb* laiocb)

{

    if (laiocb->async_context_id == get_async_context_id()) {

        qemu_laio_process_completion(s, laiocb);

    } else {

        QLIST_INSERT_HEAD(&s->completed_reqs, laiocb, node);

    }

}

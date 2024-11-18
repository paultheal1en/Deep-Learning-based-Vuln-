static void qemu_paio_submit(struct qemu_paiocb *aiocb)

{

    aiocb->ret = -EINPROGRESS;

    aiocb->active = 0;

    mutex_lock(&lock);

    if (idle_threads == 0 && cur_threads < max_threads)

        spawn_thread();

    TAILQ_INSERT_TAIL(&request_list, aiocb, node);

    mutex_unlock(&lock);

    cond_signal(&cond);

}

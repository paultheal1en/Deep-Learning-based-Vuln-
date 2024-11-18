static void thread_pool_init_one(ThreadPool *pool, AioContext *ctx)

{

    if (!ctx) {

        ctx = qemu_get_aio_context();

    }



    memset(pool, 0, sizeof(*pool));

    event_notifier_init(&pool->notifier, false);

    pool->ctx = ctx;

    qemu_mutex_init(&pool->lock);

    qemu_cond_init(&pool->check_cancel);

    qemu_cond_init(&pool->worker_stopped);

    qemu_sem_init(&pool->sem, 0);

    pool->max_threads = 64;

    pool->new_thread_bh = aio_bh_new(ctx, spawn_thread_bh_fn, pool);



    QLIST_INIT(&pool->head);

    QTAILQ_INIT(&pool->request_list);



    aio_set_event_notifier(ctx, &pool->notifier, event_notifier_ready);

}

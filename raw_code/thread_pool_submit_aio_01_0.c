BlockDriverAIOCB *thread_pool_submit_aio(ThreadPoolFunc *func, void *arg,

        BlockDriverCompletionFunc *cb, void *opaque)

{

    ThreadPool *pool = &global_pool;

    ThreadPoolElement *req;



    req = qemu_aio_get(&thread_pool_aiocb_info, NULL, cb, opaque);

    req->func = func;

    req->arg = arg;

    req->state = THREAD_QUEUED;

    req->pool = pool;



    QLIST_INSERT_HEAD(&pool->head, req, all);



    trace_thread_pool_submit(pool, req, arg);



    qemu_mutex_lock(&pool->lock);

    if (pool->idle_threads == 0 && pool->cur_threads < pool->max_threads) {

        spawn_thread(pool);

    }

    QTAILQ_INSERT_TAIL(&pool->request_list, req, reqs);

    qemu_mutex_unlock(&pool->lock);

    qemu_sem_post(&pool->sem);

    return &req->common;

}

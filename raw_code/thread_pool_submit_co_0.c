int coroutine_fn thread_pool_submit_co(ThreadPool *pool, ThreadPoolFunc *func,

                                       void *arg)

{

    ThreadPoolCo tpc = { .co = qemu_coroutine_self(), .ret = -EINPROGRESS };

    assert(qemu_in_coroutine());

    thread_pool_submit_aio(pool, func, arg, thread_pool_co_cb, &tpc);

    qemu_coroutine_yield();

    return tpc.ret;

}

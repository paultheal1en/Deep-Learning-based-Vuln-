Coroutine *qemu_coroutine_create(CoroutineEntry *entry)

{

    Coroutine *co = NULL;



    if (CONFIG_COROUTINE_POOL) {

        co = QSLIST_FIRST(&alloc_pool);

        if (!co) {

            if (release_pool_size > POOL_BATCH_SIZE) {

                /* Slow path; a good place to register the destructor, too.  */

                if (!coroutine_pool_cleanup_notifier.notify) {

                    coroutine_pool_cleanup_notifier.notify = coroutine_pool_cleanup;

                    qemu_thread_atexit_add(&coroutine_pool_cleanup_notifier);

                }



                /* This is not exact; there could be a little skew between

                 * release_pool_size and the actual size of release_pool.  But

                 * it is just a heuristic, it does not need to be perfect.

                 */

                alloc_pool_size = atomic_xchg(&release_pool_size, 0);

                QSLIST_MOVE_ATOMIC(&alloc_pool, &release_pool);

                co = QSLIST_FIRST(&alloc_pool);

            }

        }

        if (co) {

            QSLIST_REMOVE_HEAD(&alloc_pool, pool_next);

            alloc_pool_size--;

        }

    }



    if (!co) {

        co = qemu_coroutine_new();

    }



    co->entry = entry;

    QSIMPLEQ_INIT(&co->co_queue_wakeup);

    return co;

}

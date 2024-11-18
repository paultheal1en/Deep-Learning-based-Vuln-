static void thread_pool_cancel(BlockAIOCB *acb)

{

    ThreadPoolElement *elem = (ThreadPoolElement *)acb;

    ThreadPool *pool = elem->pool;



    trace_thread_pool_cancel(elem, elem->common.opaque);



    qemu_mutex_lock(&pool->lock);

    if (elem->state == THREAD_QUEUED &&

        /* No thread has yet started working on elem. we can try to "steal"

         * the item from the worker if we can get a signal from the

         * semaphore.  Because this is non-blocking, we can do it with

         * the lock taken and ensure that elem will remain THREAD_QUEUED.

         */

        qemu_sem_timedwait(&pool->sem, 0) == 0) {

        QTAILQ_REMOVE(&pool->request_list, elem, reqs);

        qemu_bh_schedule(pool->completion_bh);



        elem->state = THREAD_DONE;

        elem->ret = -ECANCELED;

    }



    qemu_mutex_unlock(&pool->lock);

}

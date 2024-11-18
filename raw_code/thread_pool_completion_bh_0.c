static void thread_pool_completion_bh(void *opaque)

{

    ThreadPool *pool = opaque;

    ThreadPoolElement *elem, *next;



restart:

    QLIST_FOREACH_SAFE(elem, &pool->head, all, next) {

        if (elem->state != THREAD_DONE) {

            continue;

        }



        trace_thread_pool_complete(pool, elem, elem->common.opaque,

                                   elem->ret);

        QLIST_REMOVE(elem, all);



        if (elem->common.cb) {

            /* Read state before ret.  */

            smp_rmb();



            /* Schedule ourselves in case elem->common.cb() calls aio_poll() to

             * wait for another request that completed at the same time.

             */

            qemu_bh_schedule(pool->completion_bh);



            elem->common.cb(elem->common.opaque, elem->ret);

            qemu_aio_unref(elem);

            goto restart;

        } else {

            qemu_aio_unref(elem);

        }

    }

}

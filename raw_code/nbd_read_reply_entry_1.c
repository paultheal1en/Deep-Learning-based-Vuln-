static coroutine_fn void nbd_read_reply_entry(void *opaque)

{

    NBDClientSession *s = opaque;

    uint64_t i;

    int ret = 0;

    Error *local_err = NULL;



    while (!s->quit) {

        assert(s->reply.handle == 0);

        ret = nbd_receive_reply(s->ioc, &s->reply, &local_err);

        if (ret < 0) {

            error_report_err(local_err);

        }

        if (ret <= 0) {

            break;

        }



        /* There's no need for a mutex on the receive side, because the

         * handler acts as a synchronization point and ensures that only

         * one coroutine is called until the reply finishes.

         */

        i = HANDLE_TO_INDEX(s, s->reply.handle);

        if (i >= MAX_NBD_REQUESTS ||

            !s->requests[i].coroutine ||

            !s->requests[i].receiving ||

            (nbd_reply_is_structured(&s->reply) && !s->info.structured_reply))

        {

            break;

        }



        /* We're woken up again by the request itself.  Note that there

         * is no race between yielding and reentering read_reply_co.  This

         * is because:

         *

         * - if the request runs on the same AioContext, it is only

         *   entered after we yield

         *

         * - if the request runs on a different AioContext, reentering

         *   read_reply_co happens through a bottom half, which can only

         *   run after we yield.

         */

        aio_co_wake(s->requests[i].coroutine);

        qemu_coroutine_yield();

    }



    s->quit = true;

    nbd_recv_coroutines_wake_all(s);

    s->read_reply_co = NULL;

}

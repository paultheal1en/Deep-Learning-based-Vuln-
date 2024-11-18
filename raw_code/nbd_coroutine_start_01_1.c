static void nbd_coroutine_start(NBDClientSession *s,

                                NBDRequest *request)

{

    /* Poor man semaphore.  The free_sema is locked when no other request

     * can be accepted, and unlocked after receiving one reply.  */

    if (s->in_flight == MAX_NBD_REQUESTS) {

        qemu_co_queue_wait(&s->free_sema, NULL);

        assert(s->in_flight < MAX_NBD_REQUESTS);

    }

    s->in_flight++;



    /* s->recv_coroutine[i] is set as soon as we get the send_lock.  */

}

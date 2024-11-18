void coroutine_fn qemu_co_mutex_unlock(CoMutex *mutex)

{

    Coroutine *self = qemu_coroutine_self();



    trace_qemu_co_mutex_unlock_entry(mutex, self);



    assert(mutex->locked);

    assert(mutex->holder == self);

    assert(qemu_in_coroutine());



    mutex->holder = NULL;

    self->locks_held--;

    if (atomic_fetch_dec(&mutex->locked) == 1) {

        /* No waiting qemu_co_mutex_lock().  Pfew, that was easy!  */

        return;

    }



    for (;;) {

        CoWaitRecord *to_wake = pop_waiter(mutex);

        unsigned our_handoff;



        if (to_wake) {

            Coroutine *co = to_wake->co;

            aio_co_wake(co);

            break;

        }



        /* Some concurrent lock() is in progress (we know this because

         * mutex->locked was >1) but it hasn't yet put itself on the wait

         * queue.  Pick a sequence number for the handoff protocol (not 0).

         */

        if (++mutex->sequence == 0) {

            mutex->sequence = 1;

        }



        our_handoff = mutex->sequence;

        atomic_mb_set(&mutex->handoff, our_handoff);

        if (!has_waiters(mutex)) {

            /* The concurrent lock has not added itself yet, so it

             * will be able to pick our handoff.

             */

            break;

        }



        /* Try to do the handoff protocol ourselves; if somebody else has

         * already taken it, however, we're done and they're responsible.

         */

        if (atomic_cmpxchg(&mutex->handoff, our_handoff, 0) != our_handoff) {

            break;

        }

    }



    trace_qemu_co_mutex_unlock_return(mutex, self);

}

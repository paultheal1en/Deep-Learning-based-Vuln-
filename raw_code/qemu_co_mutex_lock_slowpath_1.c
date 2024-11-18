static void coroutine_fn qemu_co_mutex_lock_slowpath(CoMutex *mutex)

{

    Coroutine *self = qemu_coroutine_self();

    CoWaitRecord w;

    unsigned old_handoff;



    trace_qemu_co_mutex_lock_entry(mutex, self);

    w.co = self;

    push_waiter(mutex, &w);



    /* This is the "Responsibility Hand-Off" protocol; a lock() picks from

     * a concurrent unlock() the responsibility of waking somebody up.

     */

    old_handoff = atomic_mb_read(&mutex->handoff);

    if (old_handoff &&

        has_waiters(mutex) &&

        atomic_cmpxchg(&mutex->handoff, old_handoff, 0) == old_handoff) {

        /* There can be no concurrent pops, because there can be only

         * one active handoff at a time.

         */

        CoWaitRecord *to_wake = pop_waiter(mutex);

        Coroutine *co = to_wake->co;

        if (co == self) {

            /* We got the lock ourselves!  */

            assert(to_wake == &w);

            return;

        }



        aio_co_wake(co);

    }



    qemu_coroutine_yield();

    trace_qemu_co_mutex_lock_return(mutex, self);

}

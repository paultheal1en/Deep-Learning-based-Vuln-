void qemu_cond_signal(QemuCond *cond)

{

    DWORD result;



    /*

     * Signal only when there are waiters.  cond->waiters is

     * incremented by pthread_cond_wait under the external lock,

     * so we are safe about that.

     */

    if (cond->waiters == 0) {

        return;

    }



    /*

     * Waiting threads decrement it outside the external lock, but

     * only if another thread is executing pthread_cond_broadcast and

     * has the mutex.  So, it also cannot be decremented concurrently

     * with this particular access.

     */

    cond->target = cond->waiters - 1;

    result = SignalObjectAndWait(cond->sema, cond->continue_event,

                                 INFINITE, FALSE);

    if (result == WAIT_ABANDONED || result == WAIT_FAILED) {

        error_exit(GetLastError(), __func__);

    }

}

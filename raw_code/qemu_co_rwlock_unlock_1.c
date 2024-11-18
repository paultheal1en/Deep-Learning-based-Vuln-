void qemu_co_rwlock_unlock(CoRwlock *lock)
{
    assert(qemu_in_coroutine());
    if (lock->writer) {
        lock->writer = false;
        qemu_co_queue_restart_all(&lock->queue);
    } else {
        lock->reader--;
        assert(lock->reader >= 0);
        /* Wakeup only one waiting writer */
        if (!lock->reader) {
            qemu_co_queue_next(&lock->queue);
        }
    }
}
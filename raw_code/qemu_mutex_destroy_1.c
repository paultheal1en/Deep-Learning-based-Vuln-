void qemu_mutex_destroy(QemuMutex *mutex)

{

    assert(mutex->owner == 0);

    DeleteCriticalSection(&mutex->lock);

}

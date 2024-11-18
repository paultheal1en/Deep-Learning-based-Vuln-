int qemu_mutex_trylock(QemuMutex *mutex)

{

    int owned;



    owned = TryEnterCriticalSection(&mutex->lock);

    if (owned) {

        assert(mutex->owner == 0);

        mutex->owner = GetCurrentThreadId();

    }

    return !owned;

}

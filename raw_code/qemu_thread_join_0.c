void *qemu_thread_join(QemuThread *thread)

{

    QemuThreadData *data;

    void *ret;

    HANDLE handle;



    data = thread->data;

    if (!data) {

        return NULL;

    }

    /*

     * Because multiple copies of the QemuThread can exist via

     * qemu_thread_get_self, we need to store a value that cannot

     * leak there.  The simplest, non racy way is to store the TID,

     * discard the handle that _beginthreadex gives back, and

     * get another copy of the handle here.

     */

    EnterCriticalSection(&data->cs);

    if (!data->exited) {

        handle = OpenThread(SYNCHRONIZE, FALSE, thread->tid);

        LeaveCriticalSection(&data->cs);

        WaitForSingleObject(handle, INFINITE);

        CloseHandle(handle);

    } else {

        LeaveCriticalSection(&data->cs);

    }

    ret = data->ret;

    DeleteCriticalSection(&data->cs);

    g_free(data);

    return ret;

}

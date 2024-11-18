void qemu_thread_exit(void *arg)

{

    QemuThread *thread = TlsGetValue(qemu_thread_tls_index);

    thread->ret = arg;

    CloseHandle(thread->thread);

    thread->thread = NULL;

    ExitThread(0);

}

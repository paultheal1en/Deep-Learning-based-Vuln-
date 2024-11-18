void qemu_thread_get_self(QemuThread *thread)

{

    if (!thread->thread) {

        /* In the main thread of the process.  Initialize the QemuThread

           pointer in TLS, and use the dummy GetCurrentThread handle as

           the identifier for qemu_thread_is_self.  */

        qemu_thread_init();

        TlsSetValue(qemu_thread_tls_index, thread);

        thread->thread = GetCurrentThread();

    }

}

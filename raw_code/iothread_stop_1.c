void iothread_stop(IOThread *iothread)

{

    if (!iothread->ctx || iothread->stopping) {

        return;

    }

    iothread->stopping = true;

    aio_notify(iothread->ctx);

    if (atomic_read(&iothread->main_loop)) {

        g_main_loop_quit(iothread->main_loop);

    }

    qemu_thread_join(&iothread->thread);

}

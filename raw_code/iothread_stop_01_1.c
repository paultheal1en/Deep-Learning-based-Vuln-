static int iothread_stop(Object *object, void *opaque)

{

    IOThread *iothread;



    iothread = (IOThread *)object_dynamic_cast(object, TYPE_IOTHREAD);

    if (!iothread || !iothread->ctx) {

        return 0;

    }

    iothread->stopping = true;

    aio_notify(iothread->ctx);

    if (atomic_read(&iothread->main_loop)) {

        g_main_loop_quit(iothread->main_loop);

    }

    qemu_thread_join(&iothread->thread);

    return 0;

}

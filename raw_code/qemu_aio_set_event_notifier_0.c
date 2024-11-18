void qemu_aio_set_event_notifier(EventNotifier *notifier,

                                 EventNotifierHandler *io_read,

                                 AioFlushEventNotifierHandler *io_flush)

{

    qemu_aio_set_fd_handler(event_notifier_get_fd(notifier),

                            (IOHandler *)io_read, NULL,

                            (AioFlushHandler *)io_flush, notifier);

}

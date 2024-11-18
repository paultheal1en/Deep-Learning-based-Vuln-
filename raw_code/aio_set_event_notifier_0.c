void aio_set_event_notifier(AioContext *ctx,

                            EventNotifier *notifier,

                            bool is_external,

                            EventNotifierHandler *io_read,

                            AioPollFn *io_poll)

{

    aio_set_fd_handler(ctx, event_notifier_get_fd(notifier), is_external,

                       (IOHandler *)io_read, NULL, io_poll, notifier);

}

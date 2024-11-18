void event_notifier_set_handler(EventNotifier *e,

                                EventNotifierHandler *handler)

{

    iohandler_init();

    aio_set_event_notifier(iohandler_ctx, e, false,

                           handler, NULL);

}

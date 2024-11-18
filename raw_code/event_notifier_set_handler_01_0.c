int event_notifier_set_handler(EventNotifier *e,

                               EventNotifierHandler *handler)

{

    return qemu_set_fd_handler(e->fd, (IOHandler *)handler, NULL, e);

}

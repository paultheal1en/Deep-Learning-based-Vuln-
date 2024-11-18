void event_notifier_cleanup(EventNotifier *e)

{

    close(e->fd);

}

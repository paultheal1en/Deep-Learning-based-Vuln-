void event_notifier_cleanup(EventNotifier *e)

{

    CloseHandle(e->event);


}
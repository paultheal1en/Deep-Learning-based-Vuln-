int event_notifier_test_and_clear(EventNotifier *e)

{

    uint64_t value;

    int r = read(e->fd, &value, sizeof(value));

    return r == sizeof(value);

}

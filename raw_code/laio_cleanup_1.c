void laio_cleanup(void *s_)
{
    struct qemu_laio_state *s = s_;
    event_notifier_cleanup(&s->e);
    g_free(s);
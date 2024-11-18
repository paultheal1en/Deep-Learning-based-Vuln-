static int tcp_get_msgfds(CharDriverState *chr, int *fds, int num)
{
    TCPCharDriver *s = chr->opaque;
    int to_copy = (s->read_msgfds_num < num) ? s->read_msgfds_num : num;
    if (to_copy) {
        memcpy(fds, s->read_msgfds, to_copy * sizeof(int));
        g_free(s->read_msgfds);
        s->read_msgfds = 0;
        s->read_msgfds_num = 0;
    return to_copy;
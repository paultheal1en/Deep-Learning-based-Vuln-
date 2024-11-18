static int tcp_wait_fd(int fd, int write)

{

    int ev = write ? POLLOUT : POLLIN;

    struct pollfd p = { .fd = fd, .events = ev, .revents = 0 };

    int ret;



    ret = poll(&p, 1, 100);

    return ret < 0 ? ff_neterrno() : p.revents & ev ? 0 : AVERROR(EAGAIN);

}

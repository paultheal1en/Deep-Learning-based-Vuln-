int ff_listen_bind(int fd, const struct sockaddr *addr,

                   socklen_t addrlen, int timeout)

{

    int ret;

    int reuse = 1;

    struct pollfd lp = { fd, POLLIN, 0 };

    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    ret = bind(fd, addr, addrlen);

    if (ret)

        return ff_neterrno();



    ret = listen(fd, 1);

    if (ret)

        return ff_neterrno();



    ret = poll(&lp, 1, timeout >= 0 ? timeout : -1);

    if (ret <= 0)

        return AVERROR(ETIMEDOUT);



    ret = accept(fd, NULL, NULL);

    if (ret < 0)

        return ff_neterrno();



    closesocket(fd);



    ff_socket_nonblock(ret, 1);

    return ret;

}

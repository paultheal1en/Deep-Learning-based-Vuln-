static int socket_accept(int sock)

{

    struct sockaddr_un addr;

    socklen_t addrlen;

    int ret;



    addrlen = sizeof(addr);

    do {

        ret = accept(sock, (struct sockaddr *)&addr, &addrlen);

    } while (ret == -1 && errno == EINTR);

    g_assert_no_errno(ret);

    close(sock);



    return ret;

}

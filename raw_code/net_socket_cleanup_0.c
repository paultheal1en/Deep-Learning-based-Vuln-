static void net_socket_cleanup(NetClientState *nc)

{

    NetSocketState *s = DO_UPCAST(NetSocketState, nc, nc);

    qemu_set_fd_handler(s->fd, NULL, NULL, NULL);

    close(s->fd);

}

static void net_socket_connected(int fd, Error *err, void *opaque)

{

    socket_connect_data *c = opaque;

    NetSocketState *s;

    char *addr_str = NULL;

    Error *local_error = NULL;



    addr_str = socket_address_to_string(c->saddr, &local_error);

    if (addr_str == NULL) {

        error_report_err(local_error);

        closesocket(fd);

        goto end;

    }



    s = net_socket_fd_init(c->peer, c->model, c->name, fd, true);

    if (!s) {

        closesocket(fd);

        goto end;

    }



    snprintf(s->nc.info_str, sizeof(s->nc.info_str),

             "socket: connect to %s", addr_str);



end:

    g_free(addr_str);

    socket_connect_data_free(c);

}

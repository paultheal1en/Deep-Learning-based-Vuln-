static ssize_t net_socket_receive(NetClientState *nc, const uint8_t *buf, size_t size)

{

    NetSocketState *s = DO_UPCAST(NetSocketState, nc, nc);

    uint32_t len;

    len = htonl(size);



    send_all(s->fd, (const uint8_t *)&len, sizeof(len));

    return send_all(s->fd, buf, size);

}

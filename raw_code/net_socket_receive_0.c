static void net_socket_receive(void *opaque, const uint8_t *buf, size_t size)

{

    NetSocketState *s = opaque;

    uint32_t len;

    len = htonl(size);



    send_all(s->fd, (const uint8_t *)&len, sizeof(len));

    send_all(s->fd, buf, size);

}

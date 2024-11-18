static int socket_close(void *opaque)

{

    QEMUFileSocket *s = opaque;

    closesocket(s->fd);

    g_free(s);

    return 0;

}

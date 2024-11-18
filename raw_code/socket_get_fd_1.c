static int socket_get_fd(void *opaque)

{

    QEMUFileSocket *s = opaque;



    return s->fd;

}

static int unix_close(void *opaque)

{

    QEMUFileSocket *s = opaque;

    close(s->fd);

    g_free(s);

    return 0;

}

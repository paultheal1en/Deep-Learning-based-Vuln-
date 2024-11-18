static int tcp_get_msgfd(CharDriverState *chr)

{

    TCPCharDriver *s = chr->opaque;



    return s->msgfd;

}

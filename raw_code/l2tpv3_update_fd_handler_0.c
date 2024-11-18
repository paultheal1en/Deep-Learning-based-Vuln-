static void l2tpv3_update_fd_handler(NetL2TPV3State *s)

{

    qemu_set_fd_handler2(s->fd,

                         s->read_poll ? l2tpv3_can_send : NULL,

                         s->read_poll ? net_l2tpv3_send     : NULL,

                         s->write_poll ? l2tpv3_writable : NULL,

                         s);

}

static void do_udp_write(void *arg, void *buf, int size) {

    URLContext *h = arg;

    UDPContext *s = h->priv_data;



    int ret;



    if (!(h->flags & AVIO_FLAG_NONBLOCK)) {

        ret = ff_network_wait_fd(s->udp_fd, 1);

        if (ret < 0) {

            s->circular_buffer_error = ret;

            return;

        }

    }



    if (!s->is_connected) {

        ret = sendto (s->udp_fd, buf, size, 0,

                      (struct sockaddr *) &s->dest_addr,

                      s->dest_addr_len);

    } else

        ret = send(s->udp_fd, buf, size, 0);



    s->circular_buffer_error=ret;

}

static int udp_write(URLContext *h, const uint8_t *buf, int size)

{

    UDPContext *s = h->priv_data;

    int ret;



#if HAVE_PTHREAD_CANCEL

    if (s->fifo) {

        uint8_t tmp[4];



        pthread_mutex_lock(&s->mutex);



        /*

          Return error if last tx failed.

          Here we can't know on which packet error was, but it needs to know that error exists.

        */

        if (s->circular_buffer_error<0) {

            int err=s->circular_buffer_error;

            s->circular_buffer_error=0;

            pthread_mutex_unlock(&s->mutex);

            return err;

        }



        if(av_fifo_space(s->fifo) < size + 4) {

            /* What about a partial packet tx ? */

            pthread_mutex_unlock(&s->mutex);

            return AVERROR(ENOMEM);

        }

        AV_WL32(tmp, size);

        av_fifo_generic_write(s->fifo, tmp, 4, NULL); /* size of packet */

        av_fifo_generic_write(s->fifo, (uint8_t *)buf, size, NULL); /* the data */

        pthread_cond_signal(&s->cond);

        pthread_mutex_unlock(&s->mutex);

        return size;

    }

#endif

    if (!(h->flags & AVIO_FLAG_NONBLOCK)) {

        ret = ff_network_wait_fd(s->udp_fd, 1);

        if (ret < 0)

            return ret;

    }



    if (!s->is_connected) {

        ret = sendto (s->udp_fd, buf, size, 0,

                      (struct sockaddr *) &s->dest_addr,

                      s->dest_addr_len);

    } else

        ret = send(s->udp_fd, buf, size, 0);



    return ret < 0 ? ff_neterrno() : ret;

}

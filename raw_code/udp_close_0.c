static int udp_close(URLContext *h)

{

    UDPContext *s = h->priv_data;

    int ret;



    if (s->is_multicast && (h->flags & AVIO_FLAG_READ))

        udp_leave_multicast_group(s->udp_fd, (struct sockaddr *)&s->dest_addr);

    closesocket(s->udp_fd);

    av_fifo_free(s->fifo);

#if HAVE_PTHREADS

    if (s->thread_started) {

        pthread_cancel(s->circular_buffer_thread);

        ret = pthread_join(s->circular_buffer_thread, NULL);

        if (ret != 0)

            av_log(h, AV_LOG_ERROR, "pthread_join(): %s\n", strerror(ret));

    }



    pthread_mutex_destroy(&s->mutex);

    pthread_cond_destroy(&s->cond);

#endif

    return 0;

}

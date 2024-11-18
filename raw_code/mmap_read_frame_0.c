static int mmap_read_frame(AVFormatContext *ctx, AVPacket *pkt)

{

    struct video_data *s = ctx->priv_data;

    struct v4l2_buffer buf = {

        .type   = V4L2_BUF_TYPE_VIDEO_CAPTURE,

        .memory = V4L2_MEMORY_MMAP

    };

    struct pollfd p = { .fd = s->fd, .events = POLLIN };

    int res;



    res = poll(&p, 1, s->timeout);

    if (res < 0)

        return AVERROR(errno);



    if (!(p.revents & (POLLIN | POLLERR | POLLHUP)))

        return AVERROR(EAGAIN);



    /* FIXME: Some special treatment might be needed in case of loss of signal... */

    while ((res = ioctl(s->fd, VIDIOC_DQBUF, &buf)) < 0 && (errno == EINTR));

    if (res < 0) {

        if (errno == EAGAIN) {

            pkt->size = 0;



            return AVERROR(EAGAIN);

        }

        av_log(ctx, AV_LOG_ERROR, "ioctl(VIDIOC_DQBUF): %s\n",

               strerror(errno));



        return AVERROR(errno);

    }



    if (buf.index >= s->buffers) {

        av_log(ctx, AV_LOG_ERROR, "Invalid buffer index received.\n");

        return AVERROR(EINVAL);

    }

    avpriv_atomic_int_add_and_fetch(&s->buffers_queued, -1);

    // always keep at least one buffer queued

    av_assert0(avpriv_atomic_int_get(&s->buffers_queued) >= 1);



    if (s->frame_size > 0 && buf.bytesused != s->frame_size) {

        av_log(ctx, AV_LOG_ERROR,

               "The v4l2 frame is %d bytes, but %d bytes are expected\n",

               buf.bytesused, s->frame_size);



        return AVERROR_INVALIDDATA;

    }



    /* Image is at s->buff_start[buf.index] */

    if (avpriv_atomic_int_get(&s->buffers_queued) == FFMAX(s->buffers / 8, 1)) {

        /* when we start getting low on queued buffers, fall back on copying data */

        res = av_new_packet(pkt, buf.bytesused);

        if (res < 0) {

            av_log(ctx, AV_LOG_ERROR, "Error allocating a packet.\n");

            return res;

        }

        memcpy(pkt->data, s->buf_start[buf.index], buf.bytesused);



        res = ioctl(s->fd, VIDIOC_QBUF, &buf);

        if (res < 0) {

            av_log(ctx, AV_LOG_ERROR, "ioctl(VIDIOC_QBUF)\n");

            av_free_packet(pkt);

            return AVERROR(errno);

        }

        avpriv_atomic_int_add_and_fetch(&s->buffers_queued, 1);

    } else {

        struct buff_data *buf_descriptor;



        pkt->data     = s->buf_start[buf.index];

        pkt->size     = buf.bytesused;

#if FF_API_DESTRUCT_PACKET

FF_DISABLE_DEPRECATION_WARNINGS

        pkt->destruct = dummy_release_buffer;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



        buf_descriptor = av_malloc(sizeof(struct buff_data));

        if (buf_descriptor == NULL) {

            /* Something went wrong... Since av_malloc() failed, we cannot even

             * allocate a buffer for memcpying into it

             */

            av_log(ctx, AV_LOG_ERROR, "Failed to allocate a buffer descriptor\n");

            res = ioctl(s->fd, VIDIOC_QBUF, &buf);



            return AVERROR(ENOMEM);

        }

        buf_descriptor->fd    = s->fd;

        buf_descriptor->index = buf.index;

        buf_descriptor->s     = s;



        pkt->buf = av_buffer_create(pkt->data, pkt->size, mmap_release_buffer,

                                    buf_descriptor, 0);

        if (!pkt->buf) {

            av_freep(&buf_descriptor);

            return AVERROR(ENOMEM);

        }

    }

    pkt->pts = buf.timestamp.tv_sec * INT64_C(1000000) + buf.timestamp.tv_usec;



    return s->buf_len[buf.index];

}

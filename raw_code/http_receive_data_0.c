static int http_receive_data(HTTPContext *c)

{

    HTTPContext *c1;



    if (c->buffer_end > c->buffer_ptr) {

        int len;



        len = recv(c->fd, c->buffer_ptr, c->buffer_end - c->buffer_ptr, 0);

        if (len < 0) {

            if (ff_neterrno() != FF_NETERROR(EAGAIN) &&

                ff_neterrno() != FF_NETERROR(EINTR))

                /* error : close connection */

                goto fail;

        } else if (len == 0)

            /* end of connection : close it */

            goto fail;

        else {

            c->buffer_ptr += len;

            c->data_count += len;

            update_datarate(&c->datarate, c->data_count);

        }

    }



    if (c->buffer_ptr - c->buffer >= 2 && c->data_count > FFM_PACKET_SIZE) {

        if (c->buffer[0] != 'f' ||

            c->buffer[1] != 'm') {

            http_log("Feed stream has become desynchronized -- disconnecting\n");

            goto fail;

        }

    }



    if (c->buffer_ptr >= c->buffer_end) {

        FFStream *feed = c->stream;

        /* a packet has been received : write it in the store, except

           if header */

        if (c->data_count > FFM_PACKET_SIZE) {



            //            printf("writing pos=0x%"PRIx64" size=0x%"PRIx64"\n", feed->feed_write_index, feed->feed_size);

            /* XXX: use llseek or url_seek */

            lseek(c->feed_fd, feed->feed_write_index, SEEK_SET);

            if (write(c->feed_fd, c->buffer, FFM_PACKET_SIZE) < 0) {

                http_log("Error writing to feed file: %s\n", strerror(errno));

                goto fail;

            }



            feed->feed_write_index += FFM_PACKET_SIZE;

            /* update file size */

            if (feed->feed_write_index > c->stream->feed_size)

                feed->feed_size = feed->feed_write_index;



            /* handle wrap around if max file size reached */

            if (c->stream->feed_max_size && feed->feed_write_index >= c->stream->feed_max_size)

                feed->feed_write_index = FFM_PACKET_SIZE;



            /* write index */

            ffm_write_write_index(c->feed_fd, feed->feed_write_index);



            /* wake up any waiting connections */

            for(c1 = first_http_ctx; c1 != NULL; c1 = c1->next) {

                if (c1->state == HTTPSTATE_WAIT_FEED &&

                    c1->stream->feed == c->stream->feed)

                    c1->state = HTTPSTATE_SEND_DATA;

            }

        } else {

            /* We have a header in our hands that contains useful data */

            AVFormatContext *s = NULL;

            ByteIOContext *pb;

            AVInputFormat *fmt_in;

            int i;



            url_open_buf(&pb, c->buffer, c->buffer_end - c->buffer, URL_RDONLY);

            pb->is_streamed = 1;



            /* use feed output format name to find corresponding input format */

            fmt_in = av_find_input_format(feed->fmt->name);

            if (!fmt_in)

                goto fail;



            av_open_input_stream(&s, pb, c->stream->feed_filename, fmt_in, NULL);



            /* Now we have the actual streams */

            if (s->nb_streams != feed->nb_streams) {

                av_close_input_stream(s);

                av_free(pb);

                goto fail;

            }



            for (i = 0; i < s->nb_streams; i++)

                memcpy(feed->streams[i]->codec,

                       s->streams[i]->codec, sizeof(AVCodecContext));



            av_close_input_stream(s);

            av_free(pb);

        }

        c->buffer_ptr = c->buffer;

    }



    return 0;

 fail:

    c->stream->feed_opened = 0;

    close(c->feed_fd);

    /* wake up any waiting connections to stop waiting for feed */

    for(c1 = first_http_ctx; c1 != NULL; c1 = c1->next) {

        if (c1->state == HTTPSTATE_WAIT_FEED &&

            c1->stream->feed == c->stream->feed)

            c1->state = HTTPSTATE_SEND_DATA_TRAILER;

    }

    return -1;

}

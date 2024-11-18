static int http_start_receive_data(HTTPContext *c)

{

    int fd;



    if (c->stream->feed_opened)

        return -1;



    /* Don't permit writing to this one */

    if (c->stream->readonly)

        return -1;



    /* open feed */

    fd = open(c->stream->feed_filename, O_RDWR);

    if (fd < 0) {

        http_log("Error opening feeder file: %s\n", strerror(errno));

        return -1;

    }

    c->feed_fd = fd;



    if (c->stream->truncate) {

        /* truncate feed file */

        ffm_write_write_index(c->feed_fd, FFM_PACKET_SIZE);

        ftruncate(c->feed_fd, FFM_PACKET_SIZE);

        http_log("Truncating feed file '%s'\n", c->stream->feed_filename);

    } else {

        if ((c->stream->feed_write_index = ffm_read_write_index(fd)) < 0) {

            http_log("Error reading write index from feed file: %s\n", strerror(errno));

            return -1;

        }

    }



    c->stream->feed_write_index = FFMAX(ffm_read_write_index(fd), FFM_PACKET_SIZE);

    c->stream->feed_size = lseek(fd, 0, SEEK_END);

    lseek(fd, 0, SEEK_SET);



    /* init buffer input */

    c->buffer_ptr = c->buffer;

    c->buffer_end = c->buffer + FFM_PACKET_SIZE;

    c->stream->feed_opened = 1;

    c->chunked_encoding = !!av_stristr(c->buffer, "Transfer-Encoding: chunked");

    return 0;

}

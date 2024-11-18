int init_put_byte(ByteIOContext *s,

                  unsigned char *buffer,

                  int buffer_size,

                  int write_flag,

                  void *opaque,

                  int (*read_packet)(void *opaque, uint8_t *buf, int buf_size),

                  void (*write_packet)(void *opaque, uint8_t *buf, int buf_size),

                  int (*seek)(void *opaque, offset_t offset, int whence))

{

    s->buffer = buffer;

    s->buffer_size = buffer_size;

    s->buf_ptr = buffer;

    s->write_flag = write_flag;

    if (!s->write_flag) 

        s->buf_end = buffer;

    else

        s->buf_end = buffer + buffer_size;

    s->opaque = opaque;

    s->write_packet = write_packet;

    s->read_packet = read_packet;

    s->seek = seek;

    s->pos = 0;

    s->must_flush = 0;

    s->eof_reached = 0;

    s->is_streamed = 0;

    s->max_packet_size = 0;

    s->checksum_ptr= NULL;

    s->update_checksum= NULL;

    return 0;

}

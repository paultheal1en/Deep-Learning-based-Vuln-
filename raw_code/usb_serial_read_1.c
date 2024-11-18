static void usb_serial_read(void *opaque, const uint8_t *buf, int size)

{

    USBSerialState *s = opaque;

    int first_size = RECV_BUF - s->recv_ptr;

    if (first_size > size)

        first_size = size;

    memcpy(s->recv_buf + s->recv_ptr + s->recv_used, buf, first_size);

    if (size > first_size)

        memcpy(s->recv_buf, buf + first_size, size - first_size);

    s->recv_used += size;

}

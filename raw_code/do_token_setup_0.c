static void do_token_setup(USBDevice *s, USBPacket *p)

{

    int request, value, index;



    if (p->iov.size != 8) {

        p->status = USB_RET_STALL;

        return;

    }



    usb_packet_copy(p, s->setup_buf, p->iov.size);

    p->actual_length = 0;

    s->setup_len   = (s->setup_buf[7] << 8) | s->setup_buf[6];

    s->setup_index = 0;



    request = (s->setup_buf[0] << 8) | s->setup_buf[1];

    value   = (s->setup_buf[3] << 8) | s->setup_buf[2];

    index   = (s->setup_buf[5] << 8) | s->setup_buf[4];



    if (s->setup_buf[0] & USB_DIR_IN) {

        usb_device_handle_control(s, p, request, value, index,

                                  s->setup_len, s->data_buf);

        if (p->status == USB_RET_ASYNC) {

            s->setup_state = SETUP_STATE_SETUP;

        }

        if (p->status != USB_RET_SUCCESS) {

            return;

        }



        if (p->actual_length < s->setup_len) {

            s->setup_len = p->actual_length;

        }

        s->setup_state = SETUP_STATE_DATA;

    } else {

        if (s->setup_len > sizeof(s->data_buf)) {

            fprintf(stderr,

                "usb_generic_handle_packet: ctrl buffer too small (%d > %zu)\n",

                s->setup_len, sizeof(s->data_buf));

            p->status = USB_RET_STALL;

            return;

        }

        if (s->setup_len == 0)

            s->setup_state = SETUP_STATE_ACK;

        else

            s->setup_state = SETUP_STATE_DATA;

    }



    p->actual_length = 8;

}

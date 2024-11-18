static int do_token_in(USBDevice *s, USBPacket *p)

{

    int request, value, index;

    int ret = 0;



    if (p->devep != 0)

        return s->info->handle_data(s, p);



    request = (s->setup_buf[0] << 8) | s->setup_buf[1];

    value   = (s->setup_buf[3] << 8) | s->setup_buf[2];

    index   = (s->setup_buf[5] << 8) | s->setup_buf[4];

 

    switch(s->setup_state) {

    case SETUP_STATE_ACK:

        if (!(s->setup_buf[0] & USB_DIR_IN)) {

            ret = s->info->handle_control(s, p, request, value, index,

                                          s->setup_len, s->data_buf);

            if (ret == USB_RET_ASYNC) {

                return USB_RET_ASYNC;

            }

            s->setup_state = SETUP_STATE_IDLE;

            if (ret > 0)

                return 0;

            return ret;

        }



        /* return 0 byte */

        return 0;



    case SETUP_STATE_DATA:

        if (s->setup_buf[0] & USB_DIR_IN) {

            int len = s->setup_len - s->setup_index;

            if (len > p->len)

                len = p->len;

            memcpy(p->data, s->data_buf + s->setup_index, len);

            s->setup_index += len;

            if (s->setup_index >= s->setup_len)

                s->setup_state = SETUP_STATE_ACK;

            return len;

        }



        s->setup_state = SETUP_STATE_IDLE;

        return USB_RET_STALL;



    default:

        return USB_RET_STALL;

    }

}

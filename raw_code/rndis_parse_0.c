static int rndis_parse(USBNetState *s, uint8_t *data, int length)

{

    uint32_t msg_type;

    le32 *tmp = (le32 *) data;



    msg_type = le32_to_cpup(tmp);



    switch (msg_type) {

    case RNDIS_INITIALIZE_MSG:

        s->rndis_state = RNDIS_INITIALIZED;

        return rndis_init_response(s, (rndis_init_msg_type *) data);



    case RNDIS_HALT_MSG:

        s->rndis_state = RNDIS_UNINITIALIZED;

        return 0;



    case RNDIS_QUERY_MSG:

        return rndis_query_response(s, (rndis_query_msg_type *) data, length);



    case RNDIS_SET_MSG:

        return rndis_set_response(s, (rndis_set_msg_type *) data, length);



    case RNDIS_RESET_MSG:

        rndis_clear_responsequeue(s);

        s->out_ptr = s->in_ptr = s->in_len = 0;

        return rndis_reset_response(s, (rndis_reset_msg_type *) data);



    case RNDIS_KEEPALIVE_MSG:

        /* For USB: host does this every 5 seconds */

        return rndis_keepalive_response(s, (rndis_keepalive_msg_type *) data);

    }



    return USB_RET_STALL;

}

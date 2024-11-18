static uint8_t *l2cap_bframe_out(struct bt_l2cap_conn_params_s *parm, int len)

{

    struct l2cap_chan_s *chan = (struct l2cap_chan_s *) parm;



    if (len > chan->params.remote_mtu) {

        fprintf(stderr, "%s: B-Frame for CID %04x longer than %i octets.\n",

                        __func__,

                        chan->remote_cid, chan->params.remote_mtu);

        exit(-1);

    }



    return l2cap_pdu_out(chan->l2cap, chan->remote_cid, len);

}

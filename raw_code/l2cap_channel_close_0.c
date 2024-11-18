static void l2cap_channel_close(struct l2cap_instance_s *l2cap,

                int cid, int source_cid)

{

    struct l2cap_chan_s *ch = NULL;



    /* According to Volume 3, section 6.1.1, pg 1048 of BT Core V2.0, a

     * connection in CLOSED state still responds with a L2CAP_DisconnectRsp

     * message on an L2CAP_DisconnectReq event.  */

    if (unlikely(cid < L2CAP_CID_ALLOC)) {

        l2cap_command_reject_cid(l2cap, l2cap->last_id, L2CAP_REJ_CID_INVAL,

                        cid, source_cid);

        return;

    }

    if (likely(cid >= L2CAP_CID_ALLOC && cid < L2CAP_CID_MAX))

        ch = l2cap->cid[cid];



    if (likely(ch)) {

        if (ch->remote_cid != source_cid) {

            fprintf(stderr, "%s: Ignoring a Disconnection Request with the "

                            "invalid SCID %04x.\n", __func__, source_cid);

            return;

        }



        l2cap->cid[cid] = NULL;



        ch->params.close(ch->params.opaque);

        g_free(ch);

    }



    l2cap_disconnection_response(l2cap, cid, source_cid);

}

static void l2cap_frame_in(struct l2cap_instance_s *l2cap,

                const l2cap_hdr *frame)

{

    uint16_t cid = le16_to_cpu(frame->cid);

    uint16_t len = le16_to_cpu(frame->len);



    if (unlikely(cid >= L2CAP_CID_MAX || !l2cap->cid[cid])) {

        fprintf(stderr, "%s: frame addressed to a non-existent L2CAP "

                        "channel %04x received.\n", __FUNCTION__, cid);

        return;

    }



    l2cap->cid[cid]->frame_in(l2cap->cid[cid], cid, frame, len);

}

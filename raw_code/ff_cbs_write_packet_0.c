int ff_cbs_write_packet(CodedBitstreamContext *ctx,

                        AVPacket *pkt,

                        CodedBitstreamFragment *frag)

{

    int err;



    err = ff_cbs_write_fragment_data(ctx, frag);

    if (err < 0)

        return err;



    av_new_packet(pkt, frag->data_size);

    if (err < 0)

        return err;



    memcpy(pkt->data, frag->data, frag->data_size);

    pkt->size = frag->data_size;



    return 0;

}

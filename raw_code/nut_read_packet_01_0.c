static int nut_read_packet(AVFormatContext * avf, AVPacket * pkt) {

    NUTContext * priv = avf->priv_data;

    nut_packet_t pd;

    int ret;



    while ((ret = nut_read_next_packet(priv->nut, &pd)) < 0)

        av_log(avf, AV_LOG_ERROR, " NUT error: %s\n", nut_error(-ret));



    if (ret || av_new_packet(pkt, pd.len) < 0) return -1;



    if (pd.flags & NUT_FLAG_KEY) pkt->flags |= PKT_FLAG_KEY;

    pkt->pts = pd.pts;

    pkt->stream_index = pd.stream;

    pkt->pos = url_ftell(&avf->pb);



    ret = nut_read_frame(priv->nut, &pd.len, pkt->data);



    return ret;

}

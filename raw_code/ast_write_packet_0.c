static int ast_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    AVIOContext *pb = s->pb;

    ASTMuxContext *ast = s->priv_data;

    AVCodecContext *enc = s->streams[0]->codec;

    int size = pkt->size / enc->channels;



    if (enc->frame_number == 1)

        ast->fbs = size;



    ffio_wfourcc(pb, "BLCK");

    avio_wb32(pb, size); /* Block size */



    /* padding */

    avio_wb64(pb, 0);

    avio_wb64(pb, 0);

    avio_wb64(pb, 0);



    avio_write(pb, pkt->data, pkt->size);



    return 0;

}

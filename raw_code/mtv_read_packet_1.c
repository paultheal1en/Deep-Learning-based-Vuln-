static int mtv_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    MTVDemuxContext *mtv = s->priv_data;

    ByteIOContext *pb = s->pb;

    int ret;

#if !HAVE_BIGENDIAN

    int i;

#endif



    if((url_ftell(pb) - s->data_offset + mtv->img_segment_size) % mtv->full_segment_size)

    {

        url_fskip(pb, MTV_AUDIO_PADDING_SIZE);



        ret = av_get_packet(pb, pkt, MTV_ASUBCHUNK_DATA_SIZE);

        if(ret != MTV_ASUBCHUNK_DATA_SIZE)

            return AVERROR(EIO);



        pkt->pos -= MTV_AUDIO_PADDING_SIZE;

        pkt->stream_index = AUDIO_SID;



    }else

    {

        ret = av_get_packet(pb, pkt, mtv->img_segment_size);

        if(ret != mtv->img_segment_size)

            return AVERROR(EIO);



#if !HAVE_BIGENDIAN



        /* pkt->data is GGGRRRR BBBBBGGG

         * and we need RRRRRGGG GGGBBBBB

         * for PIX_FMT_RGB565 so here we

         * just swap bytes as they come

         */



        for(i=0;i<mtv->img_segment_size/2;i++)

            *((uint16_t *)pkt->data+i) = bswap_16(*((uint16_t *)pkt->data+i));

#endif

        pkt->stream_index = VIDEO_SID;

    }



    return ret;

}

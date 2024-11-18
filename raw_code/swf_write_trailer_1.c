static int swf_write_trailer(AVFormatContext *s)

{

    SWFContext *swf = s->priv_data;

    ByteIOContext *pb = &s->pb;

    AVCodecContext *enc, *video_enc;

    int file_size, i;



    video_enc = NULL;

    for(i=0;i<s->nb_streams;i++) {

        enc = &s->streams[i]->codec;

        if (enc->codec_type == CODEC_TYPE_VIDEO)

            video_enc = enc;

    }



    put_swf_tag(s, TAG_END);

    put_swf_end_tag(s);

    

    put_flush_packet(&s->pb);



    /* patch file size and number of frames if not streamed */

    if (!url_is_streamed(&s->pb) && video_enc) {

        file_size = url_ftell(pb);

        url_fseek(pb, 4, SEEK_SET);

        put_le32(pb, file_size);

        url_fseek(pb, swf->duration_pos, SEEK_SET);

        put_le16(pb, video_enc->frame_number);

    }

    av_free(swf);

    return 0;

}

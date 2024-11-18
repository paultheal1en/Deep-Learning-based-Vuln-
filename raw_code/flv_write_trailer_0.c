static int flv_write_trailer(AVFormatContext *s)

{

    int64_t file_size;



    AVIOContext *pb = s->pb;

    FLVContext *flv = s->priv_data;

    int i;



    /* Add EOS tag */

    for (i = 0; i < s->nb_streams; i++) {

        AVCodecContext *enc = s->streams[i]->codec;

        if (enc->codec_type == AVMEDIA_TYPE_VIDEO &&

                enc->codec_id == CODEC_ID_H264) {

            put_avc_eos_tag(pb, flv->last_video_ts);

        }

    }



    file_size = avio_tell(pb);



    /* update informations */

    avio_seek(pb, flv->duration_offset, SEEK_SET);

    put_amf_double(pb, flv->duration / (double)1000);

    avio_seek(pb, flv->filesize_offset, SEEK_SET);

    put_amf_double(pb, file_size);



    avio_seek(pb, file_size, SEEK_SET);

    return 0;

}

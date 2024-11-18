static int flv_set_video_codec(AVFormatContext *s, AVStream *vstream, int flv_codecid) {

    FLVContext *flv = s->priv_data;

    AVCodecContext *vcodec = vstream->codec;

    switch(flv_codecid) {

        case FLV_CODECID_H263  : vcodec->codec_id = CODEC_ID_FLV1   ; break;

        case FLV_CODECID_SCREEN: vcodec->codec_id = CODEC_ID_FLASHSV; break;

        case FLV_CODECID_VP6A  :

            if (!flv->alpha_stream) {

                AVCodecContext *alpha_codec;

                flv->alpha_stream = av_new_stream(s, 2);

                if (flv->alpha_stream) {

                    av_set_pts_info(flv->alpha_stream, 24, 1, 1000);

                    alpha_codec = flv->alpha_stream->codec;

                    alpha_codec->codec_type = CODEC_TYPE_VIDEO;

                    alpha_codec->codec_id = CODEC_ID_VP6F;

                    alpha_codec->extradata_size = 1;

                    alpha_codec->extradata = av_malloc(1);

                }

            }

        case FLV_CODECID_VP6   : vcodec->codec_id = CODEC_ID_VP6F   ;

            if(vcodec->extradata_size != 1) {

                vcodec->extradata_size = 1;

                vcodec->extradata = av_malloc(1);

            }

            vcodec->extradata[0] = get_byte(&s->pb);

            if (flv->alpha_stream)

                flv->alpha_stream->codec->extradata[0] = vcodec->extradata[0];

            return 1; // 1 byte body size adjustment for flv_read_packet()

        default:

            av_log(s, AV_LOG_INFO, "Unsupported video codec (%x)\n", flv_codecid);

            vcodec->codec_tag = flv_codecid;

    }



    return 0;

}

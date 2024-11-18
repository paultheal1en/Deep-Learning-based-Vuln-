static void find_compressor(char * compressor_name, int len, MOVTrack *track)

{

    AVDictionaryEntry *encoder;

    int xdcam_res =  (track->par->width == 1280 && track->par->height == 720)

                  || (track->par->width == 1440 && track->par->height == 1080)

                  || (track->par->width == 1920 && track->par->height == 1080);



    if (track->mode == MODE_MOV &&

        (encoder = av_dict_get(track->st->metadata, "encoder", NULL, 0))) {

        av_strlcpy(compressor_name, encoder->value, 32);

    } else if (track->par->codec_id == AV_CODEC_ID_MPEG2VIDEO && xdcam_res) {

        int interlaced = track->par->field_order > AV_FIELD_PROGRESSIVE;

        AVStream *st = track->st;

        int rate = av_q2d(find_fps(NULL, st));

        av_strlcatf(compressor_name, len, "XDCAM");

        if (track->par->format == AV_PIX_FMT_YUV422P) {

            av_strlcatf(compressor_name, len, " HD422");

        } else if(track->par->width == 1440) {

            av_strlcatf(compressor_name, len, " HD");

        } else

            av_strlcatf(compressor_name, len, " EX");



        av_strlcatf(compressor_name, len, " %d%c", track->par->height, interlaced ? 'i' : 'p');



        av_strlcatf(compressor_name, len, "%d", rate * (interlaced + 1));

    }

}

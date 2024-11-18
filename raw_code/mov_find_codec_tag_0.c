static int mov_find_codec_tag(AVFormatContext *s, MOVTrack *track)

{

    int tag = track->enc->codec_tag;

    if (track->mode == MODE_MP4 || track->mode == MODE_PSP) {

        if (!codec_get_tag(ff_mp4_obj_type, track->enc->codec_id))

            return 0;

        if      (track->enc->codec_id == CODEC_ID_H264)      tag = MKTAG('a','v','c','1');

        else if (track->enc->codec_id == CODEC_ID_AC3)       tag = MKTAG('a','c','-','3');

        else if (track->enc->codec_id == CODEC_ID_DIRAC)     tag = MKTAG('d','r','a','c');

        else if (track->enc->codec_id == CODEC_ID_MOV_TEXT)  tag = MKTAG('t','x','3','g');

        else if (track->enc->codec_type == CODEC_TYPE_VIDEO) tag = MKTAG('m','p','4','v');

        else if (track->enc->codec_type == CODEC_TYPE_AUDIO) tag = MKTAG('m','p','4','a');

    } else if (track->mode == MODE_IPOD) {

        if (track->enc->codec_type == CODEC_TYPE_SUBTITLE &&

            (tag == MKTAG('t','x','3','g') ||

             tag == MKTAG('t','e','x','t')))

            track->tag = tag; // keep original tag

        else

            tag = codec_get_tag(codec_ipod_tags, track->enc->codec_id);

        if (!match_ext(s->filename, "m4a") && !match_ext(s->filename, "m4v"))

            av_log(s, AV_LOG_WARNING, "Warning, extension is not .m4a nor .m4v "

                   "Quicktime/Ipod might not play the file\n");

    } else if (track->mode & MODE_3GP) {

        tag = codec_get_tag(codec_3gp_tags, track->enc->codec_id);

    } else if (!tag || (track->enc->strict_std_compliance >= FF_COMPLIANCE_NORMAL &&

                        (tag == MKTAG('d','v','c','p') ||

                         track->enc->codec_id == CODEC_ID_RAWVIDEO))) {

        if (track->enc->codec_id == CODEC_ID_DVVIDEO) {

            if (track->enc->height == 480) /* NTSC */

                if  (track->enc->pix_fmt == PIX_FMT_YUV422P) tag = MKTAG('d','v','5','n');

                else                                         tag = MKTAG('d','v','c',' ');

            else if (track->enc->pix_fmt == PIX_FMT_YUV422P) tag = MKTAG('d','v','5','p');

            else if (track->enc->pix_fmt == PIX_FMT_YUV420P) tag = MKTAG('d','v','c','p');

            else                                             tag = MKTAG('d','v','p','p');

        } else if (track->enc->codec_id == CODEC_ID_RAWVIDEO) {

            tag = codec_get_tag(mov_pix_fmt_tags, track->enc->pix_fmt);

            if (!tag) // restore tag

                tag = track->enc->codec_tag;

        } else {

            if (track->enc->codec_type == CODEC_TYPE_VIDEO) {

                tag = codec_get_tag(codec_movvideo_tags, track->enc->codec_id);

                if (!tag) { // if no mac fcc found, try with Microsoft tags

                    tag = codec_get_tag(codec_bmp_tags, track->enc->codec_id);

                    if (tag)

                        av_log(s, AV_LOG_INFO, "Warning, using MS style video codec tag, "

                               "the file may be unplayable!\n");

                }

            } else if (track->enc->codec_type == CODEC_TYPE_AUDIO) {

                tag = codec_get_tag(codec_movaudio_tags, track->enc->codec_id);

                if (!tag) { // if no mac fcc found, try with Microsoft tags

                    int ms_tag = codec_get_tag(codec_wav_tags, track->enc->codec_id);

                    if (ms_tag) {

                        tag = MKTAG('m', 's', ((ms_tag >> 8) & 0xff), (ms_tag & 0xff));

                        av_log(s, AV_LOG_INFO, "Warning, using MS style audio codec tag, "

                               "the file may be unplayable!\n");

                    }

                }

            } else if (track->enc->codec_type == CODEC_TYPE_SUBTITLE) {

                tag = codec_get_tag(ff_codec_movsubtitle_tags, track->enc->codec_id);

            }

        }

    }

    return tag;

}

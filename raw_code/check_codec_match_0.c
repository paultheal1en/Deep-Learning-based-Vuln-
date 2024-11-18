int check_codec_match(AVCodecContext *ccf, AVCodecContext *ccs, int stream)

{

    int matches = 1;



#define CHECK_CODEC(x)  (ccf->x != ccs->x)

    if (CHECK_CODEC(codec_id) || CHECK_CODEC(codec_type)) {

        http_log("Codecs do not match for stream %d\n", stream);

        matches = 0;

    } else if (CHECK_CODEC(bit_rate) || CHECK_CODEC(flags)) {

        http_log("Codec bitrates do not match for stream %d\n", stream);

        matches = 0;

    } else if (ccf->codec_type == AVMEDIA_TYPE_VIDEO) {

        if (CHECK_CODEC(time_base.den) ||

            CHECK_CODEC(time_base.num) ||

            CHECK_CODEC(width) ||

            CHECK_CODEC(height)) {

            http_log("Codec width, height or framerate do not match for stream %d\n", stream);

            matches = 0;

        }

    } else if (ccf->codec_type == AVMEDIA_TYPE_AUDIO) {

        if (CHECK_CODEC(sample_rate) ||

            CHECK_CODEC(channels) ||

            CHECK_CODEC(frame_size)) {

            http_log("Codec sample_rate, channels, frame_size do not match for stream %d\n", stream);

            matches = 0;

        }

    } else {

        http_log("Unknown codec type for stream %d\n", stream);

        matches = 0;

    }



    return matches;

}

static int flic_read_header(AVFormatContext *s,

                            AVFormatParameters *ap)

{

    FlicDemuxContext *flic = s->priv_data;

    ByteIOContext *pb = &s->pb;

    unsigned char header[FLIC_HEADER_SIZE];

    AVStream *st;

    int speed;

    int magic_number;



    flic->pts = 0;



    /* load the whole header and pull out the width and height */

    if (get_buffer(pb, header, FLIC_HEADER_SIZE) != FLIC_HEADER_SIZE)

        return AVERROR(EIO);



    magic_number = AV_RL16(&header[4]);

    speed = AV_RL32(&header[0x10]);



    /* initialize the decoder streams */

    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR(ENOMEM);

    flic->video_stream_index = st->index;

    st->codec->codec_type = CODEC_TYPE_VIDEO;

    st->codec->codec_id = CODEC_ID_FLIC;

    st->codec->codec_tag = 0;  /* no fourcc */

    st->codec->width = AV_RL16(&header[0x08]);

    st->codec->height = AV_RL16(&header[0x0A]);



    if (!st->codec->width || !st->codec->height)

        return AVERROR_INVALIDDATA;



    /* send over the whole 128-byte FLIC header */

    st->codec->extradata_size = FLIC_HEADER_SIZE;

    st->codec->extradata = av_malloc(FLIC_HEADER_SIZE);

    memcpy(st->codec->extradata, header, FLIC_HEADER_SIZE);



    av_set_pts_info(st, 33, 1, 90000);



    /* Time to figure out the framerate: If there is a FLIC chunk magic

     * number at offset 0x10, assume this is from the Bullfrog game,

     * Magic Carpet. */

    if (AV_RL16(&header[0x10]) == FLIC_CHUNK_MAGIC_1) {



        flic->frame_pts_inc = FLIC_MC_PTS_INC;



        /* rewind the stream since the first chunk is at offset 12 */

        url_fseek(pb, 12, SEEK_SET);



        /* send over abbreviated FLIC header chunk */

        av_free(st->codec->extradata);

        st->codec->extradata_size = 12;

        st->codec->extradata = av_malloc(12);

        memcpy(st->codec->extradata, header, 12);



    } else if (magic_number == FLIC_FILE_MAGIC_1) {

        /*

         * in this case, the speed (n) is number of 1/70s ticks between frames:

         *

         *    pts        n * frame #

         *  --------  =  -----------  => pts = n * (90000/70) * frame #

         *   90000           70

         *

         *  therefore, the frame pts increment = n * 1285.7

         */

        flic->frame_pts_inc = speed * 1285.7;

    } else if ((magic_number == FLIC_FILE_MAGIC_2) ||

               (magic_number == FLIC_FILE_MAGIC_3)) {

        /*

         * in this case, the speed (n) is number of milliseconds between frames:

         *

         *    pts        n * frame #

         *  --------  =  -----------  => pts = n * 90 * frame #

         *   90000          1000

         *

         *  therefore, the frame pts increment = n * 90

         */

        flic->frame_pts_inc = speed * 90;

    } else {

        av_log(s, AV_LOG_INFO, "Invalid or unsupported magic chunk in file\n");

        return AVERROR_INVALIDDATA;

    }



    if (flic->frame_pts_inc == 0)

        flic->frame_pts_inc = FLIC_DEFAULT_PTS_INC;



    return 0;

}

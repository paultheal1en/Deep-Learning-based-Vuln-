static int ffm2_read_header(AVFormatContext *s)

{

    FFMContext *ffm = s->priv_data;

    AVStream *st;

    AVIOContext *pb = s->pb;

    AVCodecContext *codec;

    const AVCodecDescriptor *codec_desc;

    int ret, i;

    int f_main = 0, f_cprv = -1, f_stvi = -1, f_stau = -1;

    AVCodec *enc;

    char *buffer;



    ffm->packet_size = avio_rb32(pb);

    if (ffm->packet_size != FFM_PACKET_SIZE) {

        av_log(s, AV_LOG_ERROR, "Invalid packet size %d, expected size was %d\n",

               ffm->packet_size, FFM_PACKET_SIZE);

        ret = AVERROR_INVALIDDATA;





    ffm->write_index = avio_rb64(pb);

    /* get also filesize */

    if (pb->seekable) {

        ffm->file_size = avio_size(pb);

        if (ffm->write_index && 0)

            adjust_write_index(s);

    } else {

        ffm->file_size = (UINT64_C(1) << 63) - 1;




    while(!avio_feof(pb)) {

        unsigned id = avio_rb32(pb);

        unsigned size = avio_rb32(pb);

        int64_t next = avio_tell(pb) + size;

        char rc_eq_buf[128];



        if(!id)

            break;



        switch(id) {

        case MKBETAG('M', 'A', 'I', 'N'):

            if (f_main++) {

                ret = AVERROR(EINVAL);



            avio_rb32(pb); /* nb_streams */

            avio_rb32(pb); /* total bitrate */

            break;

        case MKBETAG('C', 'O', 'M', 'M'):

            f_cprv = f_stvi = f_stau = 0;

            st = avformat_new_stream(s, NULL);

            if (!st) {

                ret = AVERROR(ENOMEM);





            avpriv_set_pts_info(st, 64, 1, 1000000);



            codec = st->codec;

            /* generic info */

            codec->codec_id = avio_rb32(pb);

            codec_desc = avcodec_descriptor_get(codec->codec_id);

            if (!codec_desc) {

                av_log(s, AV_LOG_ERROR, "Invalid codec id: %d\n", codec->codec_id);

                codec->codec_id = AV_CODEC_ID_NONE;



            codec->codec_type = avio_r8(pb);

            if (codec->codec_type != codec_desc->type) {

                av_log(s, AV_LOG_ERROR, "Codec type mismatch: expected %d, found %d\n",

                       codec_desc->type, codec->codec_type);

                codec->codec_id = AV_CODEC_ID_NONE;

                codec->codec_type = AVMEDIA_TYPE_UNKNOWN;



            codec->bit_rate = avio_rb32(pb);

            codec->flags = avio_rb32(pb);

            codec->flags2 = avio_rb32(pb);

            codec->debug = avio_rb32(pb);

            if (codec->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {

                int size = avio_rb32(pb);

                codec->extradata = av_mallocz(size + AV_INPUT_BUFFER_PADDING_SIZE);

                if (!codec->extradata)

                    return AVERROR(ENOMEM);

                codec->extradata_size = size;

                avio_read(pb, codec->extradata, size);


            break;

        case MKBETAG('S', 'T', 'V', 'I'):

            if (f_stvi++) {

                ret = AVERROR(EINVAL);



            codec->time_base.num = avio_rb32(pb);

            codec->time_base.den = avio_rb32(pb);

            if (codec->time_base.num <= 0 || codec->time_base.den <= 0) {

                av_log(s, AV_LOG_ERROR, "Invalid time base %d/%d\n",

                       codec->time_base.num, codec->time_base.den);

                ret = AVERROR_INVALIDDATA;



            codec->width = avio_rb16(pb);

            codec->height = avio_rb16(pb);

            codec->gop_size = avio_rb16(pb);

            codec->pix_fmt = avio_rb32(pb);






            codec->qmin = avio_r8(pb);

            codec->qmax = avio_r8(pb);

            codec->max_qdiff = avio_r8(pb);

            codec->qcompress = avio_rb16(pb) / 10000.0;

            codec->qblur = avio_rb16(pb) / 10000.0;

            codec->bit_rate_tolerance = avio_rb32(pb);

            avio_get_str(pb, INT_MAX, rc_eq_buf, sizeof(rc_eq_buf));

            codec->rc_eq = av_strdup(rc_eq_buf);

            codec->rc_max_rate = avio_rb32(pb);

            codec->rc_min_rate = avio_rb32(pb);

            codec->rc_buffer_size = avio_rb32(pb);

            codec->i_quant_factor = av_int2double(avio_rb64(pb));

            codec->b_quant_factor = av_int2double(avio_rb64(pb));

            codec->i_quant_offset = av_int2double(avio_rb64(pb));

            codec->b_quant_offset = av_int2double(avio_rb64(pb));

            codec->dct_algo = avio_rb32(pb);

            codec->strict_std_compliance = avio_rb32(pb);

            codec->max_b_frames = avio_rb32(pb);

            codec->mpeg_quant = avio_rb32(pb);

            codec->intra_dc_precision = avio_rb32(pb);

            codec->me_method = avio_rb32(pb);

            codec->mb_decision = avio_rb32(pb);

            codec->nsse_weight = avio_rb32(pb);

            codec->frame_skip_cmp = avio_rb32(pb);

            codec->rc_buffer_aggressivity = av_int2double(avio_rb64(pb));

            codec->codec_tag = avio_rb32(pb);

            codec->thread_count = avio_r8(pb);

            codec->coder_type = avio_rb32(pb);

            codec->me_cmp = avio_rb32(pb);

            codec->me_subpel_quality = avio_rb32(pb);

            codec->me_range = avio_rb32(pb);

            codec->keyint_min = avio_rb32(pb);

            codec->scenechange_threshold = avio_rb32(pb);

            codec->b_frame_strategy = avio_rb32(pb);

            codec->qcompress = av_int2double(avio_rb64(pb));

            codec->qblur = av_int2double(avio_rb64(pb));

            codec->max_qdiff = avio_rb32(pb);

            codec->refs = avio_rb32(pb);

            break;

        case MKBETAG('S', 'T', 'A', 'U'):

            if (f_stau++) {

                ret = AVERROR(EINVAL);



            codec->sample_rate = avio_rb32(pb);

            codec->channels = avio_rl16(pb);

            codec->frame_size = avio_rl16(pb);

            break;

        case MKBETAG('C', 'P', 'R', 'V'):

            if (f_cprv++) {

                ret = AVERROR(EINVAL);



            enc = avcodec_find_encoder(codec->codec_id);

            if (enc && enc->priv_data_size && enc->priv_class) {

                buffer = av_malloc(size + 1);

                if (!buffer) {

                    ret = AVERROR(ENOMEM);



                avio_get_str(pb, size, buffer, size + 1);

                if ((ret = ffm_append_recommended_configuration(st, &buffer)) < 0)



            break;

        case MKBETAG('S', '2', 'V', 'I'):

            if (f_stvi++ || !size) {

                ret = AVERROR(EINVAL);



            buffer = av_malloc(size);

            if (!buffer) {

                ret = AVERROR(ENOMEM);



            avio_get_str(pb, INT_MAX, buffer, size);

            av_set_options_string(codec, buffer, "=", ",");

            if ((ret = ffm_append_recommended_configuration(st, &buffer)) < 0)


            break;

        case MKBETAG('S', '2', 'A', 'U'):

            if (f_stau++ || !size) {

                ret = AVERROR(EINVAL);



            buffer = av_malloc(size);

            if (!buffer) {

                ret = AVERROR(ENOMEM);



            avio_get_str(pb, INT_MAX, buffer, size);

            av_set_options_string(codec, buffer, "=", ",");

            if ((ret = ffm_append_recommended_configuration(st, &buffer)) < 0)


            break;


        avio_seek(pb, next, SEEK_SET);




    for (i = 0; i < s->nb_streams; i++)

        avcodec_parameters_from_context(s->streams[i]->codecpar, s->streams[i]->codec);



    /* get until end of block reached */

    while ((avio_tell(pb) % ffm->packet_size) != 0 && !pb->eof_reached)

        avio_r8(pb);



    /* init packet demux */

    ffm->packet_ptr = ffm->packet;

    ffm->packet_end = ffm->packet;

    ffm->frame_offset = 0;

    ffm->dts = 0;

    ffm->read_state = READ_HEADER;

    ffm->first_packet = 1;

    return 0;

 fail:

    ffm_close(s);

    return ret;

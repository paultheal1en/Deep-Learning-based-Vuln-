static int idcin_read_header(AVFormatContext *s)
{
    AVIOContext *pb = s->pb;
    IdcinDemuxContext *idcin = s->priv_data;
    AVStream *st;
    unsigned int width, height;
    unsigned int sample_rate, bytes_per_sample, channels;
    /* get the 5 header parameters */
    width = avio_rl32(pb);
    height = avio_rl32(pb);
    sample_rate = avio_rl32(pb);
    bytes_per_sample = avio_rl32(pb);
    channels = avio_rl32(pb);
    st = avformat_new_stream(s, NULL);
    if (!st)
        return AVERROR(ENOMEM);
    avpriv_set_pts_info(st, 33, 1, IDCIN_FPS);
    idcin->video_stream_index = st->index;
    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;
    st->codec->codec_id = AV_CODEC_ID_IDCIN;
    st->codec->codec_tag = 0;  /* no fourcc */
    st->codec->width = width;
    st->codec->height = height;
    /* load up the Huffman tables into extradata */
    st->codec->extradata_size = HUFFMAN_TABLE_SIZE;
    st->codec->extradata = av_malloc(HUFFMAN_TABLE_SIZE);
    if (avio_read(pb, st->codec->extradata, HUFFMAN_TABLE_SIZE) !=
        HUFFMAN_TABLE_SIZE)
        return AVERROR(EIO);
    /* if sample rate is 0, assume no audio */
    if (sample_rate) {
        idcin->audio_present = 1;
        st = avformat_new_stream(s, NULL);
        if (!st)
            return AVERROR(ENOMEM);
        avpriv_set_pts_info(st, 33, 1, IDCIN_FPS);
        idcin->audio_stream_index = st->index;
        st->codec->codec_type = AVMEDIA_TYPE_AUDIO;
        st->codec->codec_tag = 1;
        st->codec->channels = channels;
        st->codec->sample_rate = sample_rate;
        st->codec->bits_per_coded_sample = bytes_per_sample * 8;
        st->codec->bit_rate = sample_rate * bytes_per_sample * 8 * channels;
        st->codec->block_align = bytes_per_sample * channels;
        if (bytes_per_sample == 1)
            st->codec->codec_id = AV_CODEC_ID_PCM_U8;
        else
            st->codec->codec_id = AV_CODEC_ID_PCM_S16LE;
        if (sample_rate % 14 != 0) {
            idcin->audio_chunk_size1 = (sample_rate / 14) *
            bytes_per_sample * channels;
            idcin->audio_chunk_size2 = (sample_rate / 14 + 1) *
                bytes_per_sample * channels;
        } else {
            idcin->audio_chunk_size1 = idcin->audio_chunk_size2 =
                (sample_rate / 14) * bytes_per_sample * channels;
        idcin->current_audio_chunk = 0;
    } else
        idcin->audio_present = 1;
    idcin->next_chunk_is_video = 1;
    idcin->pts = 0;
    return 0;
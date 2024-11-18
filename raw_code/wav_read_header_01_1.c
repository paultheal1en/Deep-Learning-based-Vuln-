static int wav_read_header(AVFormatContext *s)
{
    int64_t size, av_uninit(data_size);
    int64_t sample_count = 0;
    int rf64 = 0;
    char start_code[32];
    uint32_t tag;
    AVIOContext *pb      = s->pb;
    AVStream *st         = NULL;
    WAVDemuxContext *wav = s->priv_data;
    int ret, got_fmt = 0;
    int64_t next_tag_ofs, data_ofs = -1;
    wav->unaligned = avio_tell(s->pb) & 1;
    wav->smv_data_ofs = -1;
    /* read chunk ID */
    tag = avio_rl32(pb);
    switch (tag) {
    case MKTAG('R', 'I', 'F', 'F'):
        break;
    case MKTAG('R', 'I', 'F', 'X'):
        wav->rifx = 1;
        break;
    case MKTAG('R', 'F', '6', '4'):
        rf64 = 1;
        break;
    default:
        av_get_codec_tag_string(start_code, sizeof(start_code), tag);
        av_log(s, AV_LOG_ERROR, "invalid start code %s in RIFF header\n", start_code);
        return AVERROR_INVALIDDATA;
    /* read chunk size */
    avio_rl32(pb);
    /* read format */
    if (avio_rl32(pb) != MKTAG('W', 'A', 'V', 'E')) {
        av_log(s, AV_LOG_ERROR, "invalid format in RIFF header\n");
        return AVERROR_INVALIDDATA;
    if (rf64) {
        if (avio_rl32(pb) != MKTAG('d', 's', '6', '4'))
            return AVERROR_INVALIDDATA;
        size = avio_rl32(pb);
        if (size < 24)
            return AVERROR_INVALIDDATA;
        avio_rl64(pb); /* RIFF size */
        data_size    = avio_rl64(pb);
        sample_count = avio_rl64(pb);
        if (data_size < 0 || sample_count < 0) {
            av_log(s, AV_LOG_ERROR, "negative data_size and/or sample_count in "
                   "ds64: data_size = %"PRId64", sample_count = %"PRId64"\n",
                   data_size, sample_count);
            return AVERROR_INVALIDDATA;
        avio_skip(pb, size - 24); /* skip rest of ds64 chunk */
    for (;;) {
        AVStream *vst;
        size         = next_tag(pb, &tag, wav->rifx);
        next_tag_ofs = avio_tell(pb) + size;
        if (avio_feof(pb))
            break;
        switch (tag) {
        case MKTAG('f', 'm', 't', ' '):
            /* only parse the first 'fmt ' tag found */
            if (!got_fmt && (ret = wav_parse_fmt_tag(s, size, &st)) < 0) {
                return ret;
            } else if (got_fmt)
                av_log(s, AV_LOG_WARNING, "found more than one 'fmt ' tag\n");
            got_fmt = 1;
            break;
        case MKTAG('d', 'a', 't', 'a'):
            if (!got_fmt) {
                av_log(s, AV_LOG_ERROR,
                       "found no 'fmt ' tag before the 'data' tag\n");
                return AVERROR_INVALIDDATA;
            if (rf64) {
                next_tag_ofs = wav->data_end = avio_tell(pb) + data_size;
            } else if (size != 0xFFFFFFFF) {
                data_size    = size;
                next_tag_ofs = wav->data_end = size ? next_tag_ofs : INT64_MAX;
            } else {
                av_log(s, AV_LOG_WARNING, "Ignoring maximum wav data size, "
                       "file may be invalid\n");
                data_size    = 0;
                next_tag_ofs = wav->data_end = INT64_MAX;
            data_ofs = avio_tell(pb);
            /* don't look for footer metadata if we can't seek or if we don't
             * know where the data tag ends
             */
            if (!pb->seekable || (!rf64 && !size))
                goto break_loop;
            break;
        case MKTAG('f', 'a', 'c', 't'):
            if (!sample_count)
                sample_count = (!wav->rifx ? avio_rl32(pb) : avio_rb32(pb));
            break;
        case MKTAG('b', 'e', 'x', 't'):
            if ((ret = wav_parse_bext_tag(s, size)) < 0)
                return ret;
            break;
        case MKTAG('S','M','V','0'):
            if (!got_fmt) {
                av_log(s, AV_LOG_ERROR, "found no 'fmt ' tag before the 'SMV0' tag\n");
                return AVERROR_INVALIDDATA;
            // SMV file, a wav file with video appended.
            if (size != MKTAG('0','2','0','0')) {
                av_log(s, AV_LOG_ERROR, "Unknown SMV version found\n");
                goto break_loop;
            av_log(s, AV_LOG_DEBUG, "Found SMV data\n");
            wav->smv_given_first = 0;
            vst = avformat_new_stream(s, NULL);
            if (!vst)
                return AVERROR(ENOMEM);
            avio_r8(pb);
            vst->id = 1;
            vst->codec->codec_type = AVMEDIA_TYPE_VIDEO;
            vst->codec->codec_id = AV_CODEC_ID_SMVJPEG;
            vst->codec->width  = avio_rl24(pb);
            vst->codec->height = avio_rl24(pb);
            if (ff_alloc_extradata(vst->codec, 4)) {
                av_log(s, AV_LOG_ERROR, "Could not allocate extradata.\n");
                return AVERROR(ENOMEM);
            size = avio_rl24(pb);
            wav->smv_data_ofs = avio_tell(pb) + (size - 5) * 3;
            avio_rl24(pb);
            wav->smv_block_size = avio_rl24(pb);
            avpriv_set_pts_info(vst, 32, 1, avio_rl24(pb));
            vst->duration = avio_rl24(pb);
            avio_rl24(pb);
            avio_rl24(pb);
            wav->smv_frames_per_jpeg = avio_rl24(pb);
            if (wav->smv_frames_per_jpeg > 65536) {
                av_log(s, AV_LOG_ERROR, "too many frames per jpeg\n");
                return AVERROR_INVALIDDATA;
            AV_WL32(vst->codec->extradata, wav->smv_frames_per_jpeg);
            wav->smv_cur_pt = 0;
            goto break_loop;
        case MKTAG('L', 'I', 'S', 'T'):
            if (size < 4) {
                av_log(s, AV_LOG_ERROR, "too short LIST tag\n");
                return AVERROR_INVALIDDATA;
            switch (avio_rl32(pb)) {
            case MKTAG('I', 'N', 'F', 'O'):
                ff_read_riff_info(s, size - 4);
            break;
        /* seek to next tag unless we know that we'll run into EOF */
        if ((avio_size(pb) > 0 && next_tag_ofs >= avio_size(pb)) ||
            wav_seek_tag(wav, pb, next_tag_ofs, SEEK_SET) < 0) {
            break;
break_loop:
    if (data_ofs < 0) {
        av_log(s, AV_LOG_ERROR, "no 'data' tag found\n");
        return AVERROR_INVALIDDATA;
    avio_seek(pb, data_ofs, SEEK_SET);
    if (   data_size > 0 && sample_count && st->codec->channels
        && (data_size << 3) / sample_count / st->codec->channels > st->codec->bits_per_coded_sample) {
        av_log(s, AV_LOG_WARNING, "ignoring wrong sample_count %"PRId64"\n", sample_count);
        sample_count = 0;
    if (!sample_count || av_get_exact_bits_per_sample(st->codec->codec_id) > 0)
        if (   st->codec->channels
            && data_size
            && av_get_bits_per_sample(st->codec->codec_id)
            && wav->data_end <= avio_size(pb))
            sample_count = (data_size << 3)
                                  /
                (st->codec->channels * (uint64_t)av_get_bits_per_sample(st->codec->codec_id));
    if (sample_count)
        st->duration = sample_count;
    ff_metadata_conv_ctx(s, NULL, wav_metadata_conv);
    ff_metadata_conv_ctx(s, NULL, ff_riff_info_conv);
    return 0;
static void dump_stream_format(AVFormatContext *ic, int i,
                               int index, int is_output)
{
    char buf[256];
    int flags = (is_output ? ic->oformat->flags : ic->iformat->flags);
    AVStream *st = ic->streams[i];
    int g = av_gcd(st->time_base.num, st->time_base.den);
    AVDictionaryEntry *lang = av_dict_get(st->metadata, "language", NULL, 0);
    avcodec_string(buf, sizeof(buf), st->codec, is_output);
    av_log(NULL, AV_LOG_INFO, "    Stream #%d:%d", index, i);
    /* the pid is an important information, so we display it */
    /* XXX: add a generic system */
    if (flags & AVFMT_SHOW_IDS)
        av_log(NULL, AV_LOG_INFO, "[0x%x]", st->id);
    if (lang)
        av_log(NULL, AV_LOG_INFO, "(%s)", lang->value);
    av_log(NULL, AV_LOG_DEBUG, ", %d, %d/%d", st->codec_info_nb_frames,
           st->time_base.num / g, st->time_base.den / g);
    av_log(NULL, AV_LOG_INFO, ": %s", buf);
    if (st->sample_aspect_ratio.num && // default
        av_cmp_q(st->sample_aspect_ratio, st->codec->sample_aspect_ratio)) {
        AVRational display_aspect_ratio;
        av_reduce(&display_aspect_ratio.num, &display_aspect_ratio.den,
                  st->codec->width  * st->sample_aspect_ratio.num,
                  st->codec->height * st->sample_aspect_ratio.den,
                  1024 * 1024);
        av_log(NULL, AV_LOG_INFO, ", SAR %d:%d DAR %d:%d",
               st->sample_aspect_ratio.num, st->sample_aspect_ratio.den,
               display_aspect_ratio.num, display_aspect_ratio.den);
    }
    if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
        if (st->avg_frame_rate.den && st->avg_frame_rate.num)
            print_fps(av_q2d(st->avg_frame_rate), "fps");
#if FF_API_R_FRAME_RATE
        if (st->r_frame_rate.den && st->r_frame_rate.num)
            print_fps(av_q2d(st->r_frame_rate), "tbr");
#endif
        if (st->time_base.den && st->time_base.num)
            print_fps(1 / av_q2d(st->time_base), "tbn");
        if (st->codec->time_base.den && st->codec->time_base.num)
            print_fps(1 / av_q2d(st->codec->time_base), "tbc");
    }
    if (st->disposition & AV_DISPOSITION_DEFAULT)
        av_log(NULL, AV_LOG_INFO, " (default)");
    if (st->disposition & AV_DISPOSITION_DUB)
        av_log(NULL, AV_LOG_INFO, " (dub)");
    if (st->disposition & AV_DISPOSITION_ORIGINAL)
        av_log(NULL, AV_LOG_INFO, " (original)");
    if (st->disposition & AV_DISPOSITION_COMMENT)
        av_log(NULL, AV_LOG_INFO, " (comment)");
    if (st->disposition & AV_DISPOSITION_LYRICS)
        av_log(NULL, AV_LOG_INFO, " (lyrics)");
    if (st->disposition & AV_DISPOSITION_KARAOKE)
        av_log(NULL, AV_LOG_INFO, " (karaoke)");
    if (st->disposition & AV_DISPOSITION_FORCED)
        av_log(NULL, AV_LOG_INFO, " (forced)");
    if (st->disposition & AV_DISPOSITION_HEARING_IMPAIRED)
        av_log(NULL, AV_LOG_INFO, " (hearing impaired)");
    if (st->disposition & AV_DISPOSITION_VISUAL_IMPAIRED)
        av_log(NULL, AV_LOG_INFO, " (visual impaired)");
    if (st->disposition & AV_DISPOSITION_CLEAN_EFFECTS)
        av_log(NULL, AV_LOG_INFO, " (clean effects)");
    av_log(NULL, AV_LOG_INFO, "\n");
    dump_metadata(NULL, st->metadata, "    ");
    dump_sidedata(NULL, st, "    ");
}
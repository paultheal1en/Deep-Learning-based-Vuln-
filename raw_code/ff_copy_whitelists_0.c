int ff_copy_whitelists(AVFormatContext *dst, AVFormatContext *src)

{

    av_assert0(!dst->codec_whitelist && !dst->format_whitelist);

    dst-> codec_whitelist = av_strdup(src->codec_whitelist);

    dst->format_whitelist = av_strdup(src->format_whitelist);

    if (   (src-> codec_whitelist && !dst-> codec_whitelist)

        || (src->format_whitelist && !dst->format_whitelist)) {

        av_log(dst, AV_LOG_ERROR, "Failed to duplicate whitelist\n");

        return AVERROR(ENOMEM);

    }

    return 0;

}

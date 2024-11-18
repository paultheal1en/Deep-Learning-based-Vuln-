void av_vlog(void* avcl, int level, const char *fmt, va_list vl)

{

    if(av_log_callback)

        av_log_callback(avcl, level, fmt, vl);

}

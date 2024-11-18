static int set_string(void *obj, const AVOption *o, const char *val, uint8_t **dst)

{

    av_freep(dst);

    *dst = av_strdup(val);

    return 0;

}

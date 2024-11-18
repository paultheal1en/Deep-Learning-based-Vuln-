int av_opt_get(void *obj, const char *name, int search_flags, uint8_t **out_val)

{

    void *dst, *target_obj;

    const AVOption *o = av_opt_find2(obj, name, NULL, 0, search_flags, &target_obj);

    uint8_t *bin, buf[128];

    int len, i, ret;



    if (!o || !target_obj)

        return AVERROR_OPTION_NOT_FOUND;



    dst = (uint8_t*)target_obj + o->offset;



    buf[0] = 0;

    switch (o->type) {

    case AV_OPT_TYPE_FLAGS:     ret = snprintf(buf, sizeof(buf), "0x%08X",  *(int    *)dst);break;

    case AV_OPT_TYPE_INT:       ret = snprintf(buf, sizeof(buf), "%d" ,     *(int    *)dst);break;

    case AV_OPT_TYPE_INT64:     ret = snprintf(buf, sizeof(buf), "%"PRId64, *(int64_t*)dst);break;

    case AV_OPT_TYPE_FLOAT:     ret = snprintf(buf, sizeof(buf), "%f" ,     *(float  *)dst);break;

    case AV_OPT_TYPE_DOUBLE:    ret = snprintf(buf, sizeof(buf), "%f" ,     *(double *)dst);break;

    case AV_OPT_TYPE_RATIONAL:  ret = snprintf(buf, sizeof(buf), "%d/%d",   ((AVRational*)dst)->num, ((AVRational*)dst)->den);break;

    case AV_OPT_TYPE_STRING:

        if (*(uint8_t**)dst)

            *out_val = av_strdup(*(uint8_t**)dst);

        else

            *out_val = av_strdup("");

        return 0;

    case AV_OPT_TYPE_BINARY:

        len = *(int*)(((uint8_t *)dst) + sizeof(uint8_t *));

        if ((uint64_t)len*2 + 1 > INT_MAX)

            return AVERROR(EINVAL);

        if (!(*out_val = av_malloc(len*2 + 1)))

            return AVERROR(ENOMEM);

        bin = *(uint8_t**)dst;

        for (i = 0; i < len; i++)

            snprintf(*out_val + i*2, 3, "%02X", bin[i]);

        return 0;

    default:

        return AVERROR(EINVAL);

    }



    if (ret >= sizeof(buf))

        return AVERROR(EINVAL);

    *out_val = av_strdup(buf);

    return 0;

}

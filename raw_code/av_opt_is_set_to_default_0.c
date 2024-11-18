int av_opt_is_set_to_default(void *obj, const AVOption *o)

{

    int64_t i64;

    double d, d2;

    float f;

    AVRational q;

    int ret, w, h;

    char *str;

    void *dst;



    if (!o || !obj)

        return AVERROR(EINVAL);



    dst = ((uint8_t*)obj) + o->offset;



    switch (o->type) {

    case AV_OPT_TYPE_CONST:

        return 1;

    case AV_OPT_TYPE_FLAGS:

    case AV_OPT_TYPE_PIXEL_FMT:

    case AV_OPT_TYPE_SAMPLE_FMT:

    case AV_OPT_TYPE_INT:

    case AV_OPT_TYPE_CHANNEL_LAYOUT:

    case AV_OPT_TYPE_DURATION:

    case AV_OPT_TYPE_INT64:

        read_number(o, dst, NULL, NULL, &i64);

        return o->default_val.i64 == i64;

    case AV_OPT_TYPE_STRING:

        str = *(char **)dst;

        if (str == o->default_val.str) //2 NULLs

            return 1;

        if (!str || !o->default_val.str) //1 NULL

            return 0;

        return !strcmp(str, o->default_val.str);

    case AV_OPT_TYPE_DOUBLE:

        read_number(o, dst, &d, NULL, NULL);

        return o->default_val.dbl == d;

    case AV_OPT_TYPE_FLOAT:

        read_number(o, dst, &d, NULL, NULL);

        f = o->default_val.dbl;

        d2 = f;

        return d2 == d;

    case AV_OPT_TYPE_RATIONAL:

        q = av_d2q(o->default_val.dbl, INT_MAX);

        return !av_cmp_q(*(AVRational*)dst, q);

    case AV_OPT_TYPE_BINARY: {

        struct {

            uint8_t *data;

            int size;

        } tmp = {0};

        int opt_size = *(int *)((void **)dst + 1);

        void *opt_ptr = *(void **)dst;

        if (!opt_ptr && (!o->default_val.str || !strlen(o->default_val.str)))

            return 1;

        if (opt_ptr && o->default_val.str && !strlen(o->default_val.str))

            return 0;

        if (opt_size != strlen(o->default_val.str) / 2)

            return 0;

        ret = set_string_binary(NULL, NULL, o->default_val.str, &tmp.data);

        if (!ret)

            ret = !memcmp(opt_ptr, tmp.data, tmp.size);

        av_free(tmp.data);

        return ret;

    }

    case AV_OPT_TYPE_DICT:

        /* Binary and dict have not default support yet. Any pointer is not default. */

        return !!(*(void **)dst);

    case AV_OPT_TYPE_IMAGE_SIZE:

        if (!o->default_val.str || !strcmp(o->default_val.str, "none"))

            w = h = 0;

        else if ((ret = av_parse_video_size(&w, &h, o->default_val.str)) < 0)

            return ret;

        return (w == *(int *)dst) && (h == *((int *)dst+1));

    case AV_OPT_TYPE_VIDEO_RATE:

        q = (AVRational){0, 0};

        if (o->default_val.str)

            av_parse_video_rate(&q, o->default_val.str);

        return !av_cmp_q(*(AVRational*)dst, q);

    case AV_OPT_TYPE_COLOR: {

        uint8_t color[4] = {0, 0, 0, 0};

        if (o->default_val.str)

            av_parse_color(color, o->default_val.str, -1, NULL);

        return !memcmp(color, dst, sizeof(color));

    }

    default:

        av_log(obj, AV_LOG_WARNING, "Not supported option type: %d, option name: %s\n", o->type, o->name);

        break;

    }

    return AVERROR_PATCHWELCOME;

}

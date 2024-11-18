int av_opt_set(void *obj, const char *name, const char *val, int search_flags)

{

    int ret = 0;

    void *dst, *target_obj;

    const AVOption *o = av_opt_find2(obj, name, NULL, 0, search_flags, &target_obj);

    if (!o || !target_obj)

        return AVERROR_OPTION_NOT_FOUND;

    if (!val && (o->type != AV_OPT_TYPE_STRING &&

                 o->type != AV_OPT_TYPE_PIXEL_FMT && o->type != AV_OPT_TYPE_SAMPLE_FMT &&

                 o->type != AV_OPT_TYPE_IMAGE_SIZE && o->type != AV_OPT_TYPE_VIDEO_RATE &&

                 o->type != AV_OPT_TYPE_DURATION && o->type != AV_OPT_TYPE_COLOR &&

                 o->type != AV_OPT_TYPE_CHANNEL_LAYOUT && o->type != AV_OPT_TYPE_BOOL))

        return AVERROR(EINVAL);



    if (o->flags & AV_OPT_FLAG_READONLY)

        return AVERROR(EINVAL);



    dst = ((uint8_t *)target_obj) + o->offset;

    switch (o->type) {

    case AV_OPT_TYPE_BOOL:

        return set_string_bool(obj, o, val, dst);

    case AV_OPT_TYPE_STRING:

        return set_string(obj, o, val, dst);

    case AV_OPT_TYPE_BINARY:

        return set_string_binary(obj, o, val, dst);

    case AV_OPT_TYPE_FLAGS:

    case AV_OPT_TYPE_INT:

    case AV_OPT_TYPE_INT64:

    case AV_OPT_TYPE_FLOAT:

    case AV_OPT_TYPE_DOUBLE:

    case AV_OPT_TYPE_RATIONAL:

        return set_string_number(obj, target_obj, o, val, dst);

    case AV_OPT_TYPE_IMAGE_SIZE:

        return set_string_image_size(obj, o, val, dst);

    case AV_OPT_TYPE_VIDEO_RATE:

        return set_string_video_rate(obj, o, val, dst);

    case AV_OPT_TYPE_PIXEL_FMT:

        return set_string_pixel_fmt(obj, o, val, dst);

    case AV_OPT_TYPE_SAMPLE_FMT:

        return set_string_sample_fmt(obj, o, val, dst);

    case AV_OPT_TYPE_DURATION:

        if (!val) {

            *(int64_t *)dst = 0;

            return 0;

        } else {

            if ((ret = av_parse_time(dst, val, 1)) < 0)

                av_log(obj, AV_LOG_ERROR, "Unable to parse option value \"%s\" as duration\n", val);

            return ret;

        }

        break;

    case AV_OPT_TYPE_COLOR:

        return set_string_color(obj, o, val, dst);

    case AV_OPT_TYPE_CHANNEL_LAYOUT:

        if (!val || !strcmp(val, "none")) {

            *(int64_t *)dst = 0;

        } else {

            int64_t cl = av_get_channel_layout(val);

            if (!cl) {

                av_log(obj, AV_LOG_ERROR, "Unable to parse option value \"%s\" as channel layout\n", val);

                ret = AVERROR(EINVAL);

            }

            *(int64_t *)dst = cl;

            return ret;

        }

        break;

    }



    av_log(obj, AV_LOG_ERROR, "Invalid option type.\n");

    return AVERROR(EINVAL);

}

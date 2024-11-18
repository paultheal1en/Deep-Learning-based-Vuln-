static enum AVPixelFormat get_format(AVCodecContext *s, const enum AVPixelFormat *pix_fmts)

{

    InputStream *ist = s->opaque;

    const enum AVPixelFormat *p;

    int ret;



    for (p = pix_fmts; *p != -1; p++) {

        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(*p);

        const HWAccel *hwaccel;



        if (!(desc->flags & AV_PIX_FMT_FLAG_HWACCEL))

            break;



        hwaccel = get_hwaccel(*p, ist->hwaccel_id);

        if (!hwaccel ||

            (ist->active_hwaccel_id && ist->active_hwaccel_id != hwaccel->id) ||

            (ist->hwaccel_id != HWACCEL_AUTO && ist->hwaccel_id != hwaccel->id))

            continue;



        ret = hwaccel->init(s);

        if (ret < 0) {

            if (ist->hwaccel_id == hwaccel->id) {

                av_log(NULL, AV_LOG_FATAL,

                       "%s hwaccel requested for input stream #%d:%d, "

                       "but cannot be initialized.\n", hwaccel->name,

                       ist->file_index, ist->st->index);

                return AV_PIX_FMT_NONE;

            }

            continue;

        }



        if (ist->hw_frames_ctx) {

            s->hw_frames_ctx = av_buffer_ref(ist->hw_frames_ctx);

            if (!s->hw_frames_ctx)

                return AV_PIX_FMT_NONE;

        }



        ist->active_hwaccel_id = hwaccel->id;

        ist->hwaccel_pix_fmt   = *p;

        break;

    }



    return *p;

}

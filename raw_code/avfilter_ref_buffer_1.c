AVFilterBufferRef *avfilter_ref_buffer(AVFilterBufferRef *ref, int pmask)
{
    AVFilterBufferRef *ret = av_malloc(sizeof(AVFilterBufferRef));
    if (!ret)
        return NULL;
    *ret = *ref;
    if (ref->type == AVMEDIA_TYPE_VIDEO) {
        ret->video = av_malloc(sizeof(AVFilterBufferRefVideoProps));
        if (!ret->video) {
            av_free(ret);
            return NULL;
        }
        copy_video_props(ret->video, ref->video);
        ret->extended_data = ret->data;
    } else if (ref->type == AVMEDIA_TYPE_AUDIO) {
        ret->audio = av_malloc(sizeof(AVFilterBufferRefAudioProps));
        if (!ret->audio) {
            av_free(ret);
            return NULL;
        }
        *ret->audio = *ref->audio;
        if (ref->extended_data && ref->extended_data != ref->data) {
            int nb_channels = av_get_channel_layout_nb_channels(ref->audio->channel_layout);
            if (!(ret->extended_data = av_malloc(sizeof(*ret->extended_data) *
                                                 nb_channels))) {
                av_freep(&ret->audio);
                av_freep(&ret);
                return NULL;
            }
            memcpy(ret->extended_data, ref->extended_data,
                   sizeof(*ret->extended_data) * nb_channels);
        } else
            ret->extended_data = ret->data;
    }
    ret->perms &= pmask;
    ret->buf->refcount ++;
    return ret;
}
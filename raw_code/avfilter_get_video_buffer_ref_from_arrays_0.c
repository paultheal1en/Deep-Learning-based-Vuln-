avfilter_get_video_buffer_ref_from_arrays(uint8_t *data[4], int linesize[4], int perms,

                                          int w, int h, enum PixelFormat format)

{

    AVFilterBuffer *pic = av_mallocz(sizeof(AVFilterBuffer));

    AVFilterBufferRef *picref = av_mallocz(sizeof(AVFilterBufferRef));



    if (!pic || !picref)

        goto fail;



    picref->buf = pic;

    picref->buf->free = ff_avfilter_default_free_buffer;

    if (!(picref->video = av_mallocz(sizeof(AVFilterBufferRefVideoProps))))

        goto fail;



    picref->video->w = w;

    picref->video->h = h;



    /* make sure the buffer gets read permission or it's useless for output */

    picref->perms = perms | AV_PERM_READ;



    pic->refcount = 1;

    picref->type = AVMEDIA_TYPE_VIDEO;

    picref->format = format;



    memcpy(pic->data,        data,          sizeof(pic->data));

    memcpy(pic->linesize,    linesize,      sizeof(pic->linesize));

    memcpy(picref->data,     pic->data,     sizeof(picref->data));

    memcpy(picref->linesize, pic->linesize, sizeof(picref->linesize));



    return picref;



fail:

    if (picref && picref->video)

        av_free(picref->video);

    av_free(picref);

    av_free(pic);

    return NULL;

}

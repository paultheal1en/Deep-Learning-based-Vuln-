int avcodec_copy_context(AVCodecContext *dest, const AVCodecContext *src)

{

    const AVCodec *orig_codec = dest->codec;

    uint8_t *orig_priv_data = dest->priv_data;



    if (avcodec_is_open(dest)) { // check that the dest context is uninitialized

        av_log(dest, AV_LOG_ERROR,

               "Tried to copy AVCodecContext %p into already-initialized %p\n",

               src, dest);

        return AVERROR(EINVAL);

    }



    av_opt_free(dest);

    av_free(dest->priv_data);



    memcpy(dest, src, sizeof(*dest));



    dest->priv_data       = orig_priv_data;

    dest->codec           = orig_codec;



    /* set values specific to opened codecs back to their default state */

    dest->slice_offset    = NULL;

    dest->hwaccel         = NULL;

    dest->internal        = NULL;



    /* reallocate values that should be allocated separately */

    dest->rc_eq           = NULL;

    dest->extradata       = NULL;

    dest->intra_matrix    = NULL;

    dest->inter_matrix    = NULL;

    dest->rc_override     = NULL;

    dest->subtitle_header = NULL;

    if (src->rc_eq) {

        dest->rc_eq = av_strdup(src->rc_eq);

        if (!dest->rc_eq)

            return AVERROR(ENOMEM);

    }



#define alloc_and_copy_or_fail(obj, size, pad) \

    if (src->obj && size > 0) { \

        dest->obj = av_malloc(size + pad); \

        if (!dest->obj) \

            goto fail; \

        memcpy(dest->obj, src->obj, size); \

        if (pad) \

            memset(((uint8_t *) dest->obj) + size, 0, pad); \

    }

    alloc_and_copy_or_fail(extradata,    src->extradata_size,

                           FF_INPUT_BUFFER_PADDING_SIZE);

    alloc_and_copy_or_fail(intra_matrix, 64 * sizeof(int16_t), 0);

    alloc_and_copy_or_fail(inter_matrix, 64 * sizeof(int16_t), 0);

    alloc_and_copy_or_fail(rc_override,  src->rc_override_count * sizeof(*src->rc_override), 0);

    alloc_and_copy_or_fail(subtitle_header, src->subtitle_header_size, 1);

    dest->subtitle_header_size = src->subtitle_header_size;

#undef alloc_and_copy_or_fail



    return 0;



fail:

    av_freep(&dest->rc_override);

    av_freep(&dest->intra_matrix);

    av_freep(&dest->inter_matrix);

    av_freep(&dest->extradata);

    av_freep(&dest->rc_eq);

    return AVERROR(ENOMEM);

}

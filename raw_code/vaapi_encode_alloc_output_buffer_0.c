static AVBufferRef *vaapi_encode_alloc_output_buffer(void *opaque,

                                                     int size)

{

    AVCodecContext   *avctx = opaque;

    VAAPIEncodeContext *ctx = avctx->priv_data;

    VABufferID buffer_id;

    VAStatus vas;

    AVBufferRef *ref;



    // The output buffer size is fixed, so it needs to be large enough

    // to hold the largest possible compressed frame.  We assume here

    // that the uncompressed frame plus some header data is an upper

    // bound on that.

    vas = vaCreateBuffer(ctx->hwctx->display, ctx->va_context,

                         VAEncCodedBufferType,

                         3 * ctx->aligned_width * ctx->aligned_height +

                         (1 << 16), 1, 0, &buffer_id);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "Failed to create bitstream "

               "output buffer: %d (%s).\n", vas, vaErrorStr(vas));

        return NULL;

    }



    av_log(avctx, AV_LOG_DEBUG, "Allocated output buffer %#x\n", buffer_id);



    ref = av_buffer_create((uint8_t*)(uintptr_t)buffer_id,

                           sizeof(buffer_id),

                           &vaapi_encode_free_output_buffer,

                           avctx, AV_BUFFER_FLAG_READONLY);

    if (!ref) {

        vaDestroyBuffer(ctx->hwctx->display, buffer_id);

        return NULL;

    }



    return ref;

}

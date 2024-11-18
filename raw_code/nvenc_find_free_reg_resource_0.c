static int nvenc_find_free_reg_resource(AVCodecContext *avctx)

{

    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;

    NV_ENCODE_API_FUNCTION_LIST *p_nvenc = &dl_fn->nvenc_funcs;



    int i;



    if (ctx->nb_registered_frames == FF_ARRAY_ELEMS(ctx->registered_frames)) {

        for (i = 0; i < ctx->nb_registered_frames; i++) {

            if (!ctx->registered_frames[i].mapped) {

                if (ctx->registered_frames[i].regptr) {

                    p_nvenc->nvEncUnregisterResource(ctx->nvencoder,

                                                ctx->registered_frames[i].regptr);

                    ctx->registered_frames[i].regptr = NULL;

                }

                return i;

            }

        }

    } else {

        return ctx->nb_registered_frames++;

    }



    av_log(avctx, AV_LOG_ERROR, "Too many registered CUDA frames\n");

    return AVERROR(ENOMEM);

}

static int nvenc_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

    const AVFrame *frame, int *got_packet)

{

    NVENCSTATUS nv_status;

    NvencOutputSurface *tmpoutsurf;

    int res, i = 0;



    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;

    NV_ENCODE_API_FUNCTION_LIST *p_nvenc = &dl_fn->nvenc_funcs;



    NV_ENC_PIC_PARAMS pic_params = { 0 };

    pic_params.version = NV_ENC_PIC_PARAMS_VER;



    if (frame) {

        NV_ENC_LOCK_INPUT_BUFFER lockBufferParams = { 0 };

        NvencInputSurface *inSurf = NULL;



        for (i = 0; i < ctx->max_surface_count; ++i) {

            if (!ctx->input_surfaces[i].lockCount) {

                inSurf = &ctx->input_surfaces[i];

                break;

            }

        }



        av_assert0(inSurf);



        inSurf->lockCount = 1;



        lockBufferParams.version = NV_ENC_LOCK_INPUT_BUFFER_VER;

        lockBufferParams.inputBuffer = inSurf->input_surface;



        nv_status = p_nvenc->nvEncLockInputBuffer(ctx->nvencoder, &lockBufferParams);

        if (nv_status != NV_ENC_SUCCESS) {

            av_log(avctx, AV_LOG_ERROR, "Failed locking nvenc input buffer\n");

            return 0;

        }



        if (avctx->pix_fmt == AV_PIX_FMT_YUV420P) {

            uint8_t *buf = lockBufferParams.bufferDataPtr;



            av_image_copy_plane(buf, lockBufferParams.pitch,

                frame->data[0], frame->linesize[0],

                avctx->width, avctx->height);



            buf += inSurf->height * lockBufferParams.pitch;



            av_image_copy_plane(buf, lockBufferParams.pitch >> 1,

                frame->data[2], frame->linesize[2],

                avctx->width >> 1, avctx->height >> 1);



            buf += (inSurf->height * lockBufferParams.pitch) >> 2;



            av_image_copy_plane(buf, lockBufferParams.pitch >> 1,

                frame->data[1], frame->linesize[1],

                avctx->width >> 1, avctx->height >> 1);

        } else if (avctx->pix_fmt == AV_PIX_FMT_NV12) {

            uint8_t *buf = lockBufferParams.bufferDataPtr;



            av_image_copy_plane(buf, lockBufferParams.pitch,

                frame->data[0], frame->linesize[0],

                avctx->width, avctx->height);



            buf += inSurf->height * lockBufferParams.pitch;



            av_image_copy_plane(buf, lockBufferParams.pitch,

                frame->data[1], frame->linesize[1],

                avctx->width, avctx->height >> 1);

        } else if (avctx->pix_fmt == AV_PIX_FMT_YUV444P) {

            uint8_t *buf = lockBufferParams.bufferDataPtr;



            av_image_copy_plane(buf, lockBufferParams.pitch,

                frame->data[0], frame->linesize[0],

                avctx->width, avctx->height);



            buf += inSurf->height * lockBufferParams.pitch;



            av_image_copy_plane(buf, lockBufferParams.pitch,

                frame->data[1], frame->linesize[1],

                avctx->width, avctx->height);



            buf += inSurf->height * lockBufferParams.pitch;



            av_image_copy_plane(buf, lockBufferParams.pitch,

                frame->data[2], frame->linesize[2],

                avctx->width, avctx->height);

        } else {

            av_log(avctx, AV_LOG_FATAL, "Invalid pixel format!\n");

            return AVERROR(EINVAL);

        }



        nv_status = p_nvenc->nvEncUnlockInputBuffer(ctx->nvencoder, inSurf->input_surface);

        if (nv_status != NV_ENC_SUCCESS) {

            av_log(avctx, AV_LOG_FATAL, "Failed unlocking input buffer!\n");

            return AVERROR_EXTERNAL;

        }



        for (i = 0; i < ctx->max_surface_count; ++i)

            if (!ctx->output_surfaces[i].busy)

                break;



        if (i == ctx->max_surface_count) {

            inSurf->lockCount = 0;

            av_log(avctx, AV_LOG_FATAL, "No free output surface found!\n");

            return AVERROR_EXTERNAL;

        }



        ctx->output_surfaces[i].input_surface = inSurf;



        pic_params.inputBuffer = inSurf->input_surface;

        pic_params.bufferFmt = inSurf->format;

        pic_params.inputWidth = avctx->width;

        pic_params.inputHeight = avctx->height;

        pic_params.outputBitstream = ctx->output_surfaces[i].output_surface;

        pic_params.completionEvent = 0;



        if (avctx->flags & AV_CODEC_FLAG_INTERLACED_DCT) {

            if (frame->top_field_first) {

                pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM;

            } else {

                pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP;

            }

        } else {

            pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

        }



        pic_params.encodePicFlags = 0;

        pic_params.inputTimeStamp = frame->pts;

        pic_params.inputDuration = 0;

        switch (avctx->codec->id) {

        case AV_CODEC_ID_H264:

          pic_params.codecPicParams.h264PicParams.sliceMode = ctx->encode_config.encodeCodecConfig.h264Config.sliceMode;

          pic_params.codecPicParams.h264PicParams.sliceModeData = ctx->encode_config.encodeCodecConfig.h264Config.sliceModeData;

          break;

        case AV_CODEC_ID_H265:

          pic_params.codecPicParams.hevcPicParams.sliceMode = ctx->encode_config.encodeCodecConfig.hevcConfig.sliceMode;

          pic_params.codecPicParams.hevcPicParams.sliceModeData = ctx->encode_config.encodeCodecConfig.hevcConfig.sliceModeData;

          break;

        default:

          av_log(avctx, AV_LOG_ERROR, "Unknown codec name\n");

          return AVERROR(EINVAL);

        }



        res = timestamp_queue_enqueue(&ctx->timestamp_list, frame->pts);



        if (res)

            return res;

    } else {

        pic_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;

    }



    nv_status = p_nvenc->nvEncEncodePicture(ctx->nvencoder, &pic_params);



    if (frame && nv_status == NV_ENC_ERR_NEED_MORE_INPUT) {

        res = out_surf_queue_enqueue(&ctx->output_surface_queue, &ctx->output_surfaces[i]);



        if (res)

            return res;



        ctx->output_surfaces[i].busy = 1;

    }



    if (nv_status != NV_ENC_SUCCESS && nv_status != NV_ENC_ERR_NEED_MORE_INPUT) {

        av_log(avctx, AV_LOG_ERROR, "EncodePicture failed!\n");

        return AVERROR_EXTERNAL;

    }



    if (nv_status != NV_ENC_ERR_NEED_MORE_INPUT) {

        while (ctx->output_surface_queue.count) {

            tmpoutsurf = out_surf_queue_dequeue(&ctx->output_surface_queue);

            res = out_surf_queue_enqueue(&ctx->output_surface_ready_queue, tmpoutsurf);



            if (res)

                return res;

        }



        if (frame) {

            res = out_surf_queue_enqueue(&ctx->output_surface_ready_queue, &ctx->output_surfaces[i]);



            if (res)

                return res;



            ctx->output_surfaces[i].busy = 1;

        }

    }



    if (ctx->output_surface_ready_queue.count && (!frame || ctx->output_surface_ready_queue.count + ctx->output_surface_queue.count >= ctx->buffer_delay)) {

        tmpoutsurf = out_surf_queue_dequeue(&ctx->output_surface_ready_queue);



        res = process_output_surface(avctx, pkt, tmpoutsurf);



        if (res)

            return res;



        tmpoutsurf->busy = 0;

        av_assert0(tmpoutsurf->input_surface->lockCount);

        tmpoutsurf->input_surface->lockCount--;



        *got_packet = 1;

    } else {

        *got_packet = 0;

    }



    return 0;

}

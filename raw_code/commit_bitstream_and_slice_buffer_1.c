static int commit_bitstream_and_slice_buffer(AVCodecContext *avctx,

                                             DECODER_BUFFER_DESC *bs,

                                             DECODER_BUFFER_DESC *sc)

{

    const H264Context *h = avctx->priv_data;

    const unsigned mb_count = h->mb_width * h->mb_height;

    AVDXVAContext *ctx = avctx->hwaccel_context;

    const H264Picture *current_picture = h->cur_pic_ptr;

    struct dxva2_picture_context *ctx_pic = current_picture->hwaccel_picture_private;

    DXVA_Slice_H264_Short *slice = NULL;

    void     *dxva_data_ptr;

    uint8_t  *dxva_data, *current, *end;

    unsigned dxva_size;

    void     *slice_data;

    unsigned slice_size;

    unsigned padding;

    unsigned i;

    unsigned type;



    /* Create an annex B bitstream buffer with only slice NAL and finalize slice */

#if CONFIG_D3D11VA

    if (avctx->pix_fmt == AV_PIX_FMT_D3D11VA_VLD) {

        type = D3D11_VIDEO_DECODER_BUFFER_BITSTREAM;

        if (FAILED(ID3D11VideoContext_GetDecoderBuffer(D3D11VA_CONTEXT(ctx)->video_context,

                                                       D3D11VA_CONTEXT(ctx)->decoder,

                                                       type,

                                                       &dxva_size, &dxva_data_ptr)))

            return -1;

    }

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD) {

        type = DXVA2_BitStreamDateBufferType;

        if (FAILED(IDirectXVideoDecoder_GetBuffer(DXVA2_CONTEXT(ctx)->decoder,

                                                  type,

                                                  &dxva_data_ptr, &dxva_size)))

            return -1;

    }

#endif



    dxva_data = dxva_data_ptr;

    current = dxva_data;

    end = dxva_data + dxva_size;



    for (i = 0; i < ctx_pic->slice_count; i++) {

        static const uint8_t start_code[] = { 0, 0, 1 };

        static const unsigned start_code_size = sizeof(start_code);

        unsigned position, size;



        assert(offsetof(DXVA_Slice_H264_Short, BSNALunitDataLocation) ==

               offsetof(DXVA_Slice_H264_Long,  BSNALunitDataLocation));

        assert(offsetof(DXVA_Slice_H264_Short, SliceBytesInBuffer) ==

               offsetof(DXVA_Slice_H264_Long,  SliceBytesInBuffer));



        if (is_slice_short(avctx, ctx))

            slice = &ctx_pic->slice_short[i];

        else

            slice = (DXVA_Slice_H264_Short*)&ctx_pic->slice_long[i];



        position = slice->BSNALunitDataLocation;

        size     = slice->SliceBytesInBuffer;

        if (start_code_size + size > end - current) {

            av_log(avctx, AV_LOG_ERROR, "Failed to build bitstream");

            break;

        }



        slice->BSNALunitDataLocation = current - dxva_data;

        slice->SliceBytesInBuffer    = start_code_size + size;



        if (!is_slice_short(avctx, ctx)) {

            DXVA_Slice_H264_Long *slice_long = (DXVA_Slice_H264_Long*)slice;

            if (i < ctx_pic->slice_count - 1)

                slice_long->NumMbsForSlice =

                    slice_long[1].first_mb_in_slice - slice_long[0].first_mb_in_slice;

            else

                slice_long->NumMbsForSlice = mb_count - slice_long->first_mb_in_slice;

        }



        memcpy(current, start_code, start_code_size);

        current += start_code_size;



        memcpy(current, &ctx_pic->bitstream[position], size);

        current += size;

    }

    padding = FFMIN(128 - ((current - dxva_data) & 127), end - current);

    if (slice && padding > 0) {

        memset(current, 0, padding);

        current += padding;



        slice->SliceBytesInBuffer += padding;

    }

#if CONFIG_D3D11VA

    if (avctx->pix_fmt == AV_PIX_FMT_D3D11VA_VLD)

        if (FAILED(ID3D11VideoContext_ReleaseDecoderBuffer(D3D11VA_CONTEXT(ctx)->video_context, D3D11VA_CONTEXT(ctx)->decoder, type)))

            return -1;

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD)

        if (FAILED(IDirectXVideoDecoder_ReleaseBuffer(DXVA2_CONTEXT(ctx)->decoder, type)))

            return -1;

#endif

    if (i < ctx_pic->slice_count)

        return -1;



#if CONFIG_D3D11VA

    if (avctx->pix_fmt == AV_PIX_FMT_D3D11VA_VLD) {

        D3D11_VIDEO_DECODER_BUFFER_DESC *dsc11 = bs;

        memset(dsc11, 0, sizeof(*dsc11));

        dsc11->BufferType           = type;

        dsc11->DataSize             = current - dxva_data;

        dsc11->NumMBsInBuffer       = mb_count;



        type = D3D11_VIDEO_DECODER_BUFFER_SLICE_CONTROL;

    }

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD) {

        DXVA2_DecodeBufferDesc *dsc2 = bs;

        memset(dsc2, 0, sizeof(*dsc2));

        dsc2->CompressedBufferType = type;

        dsc2->DataSize             = current - dxva_data;

        dsc2->NumMBsInBuffer       = mb_count;



        type = DXVA2_SliceControlBufferType;

    }

#endif



    if (is_slice_short(avctx, ctx)) {

        slice_data = ctx_pic->slice_short;

        slice_size = ctx_pic->slice_count * sizeof(*ctx_pic->slice_short);

    } else {

        slice_data = ctx_pic->slice_long;

        slice_size = ctx_pic->slice_count * sizeof(*ctx_pic->slice_long);

    }

    assert((bs->DataSize & 127) == 0);

    return ff_dxva2_commit_buffer(avctx, ctx, sc,

                                  type,

                                  slice_data, slice_size, mb_count);

}

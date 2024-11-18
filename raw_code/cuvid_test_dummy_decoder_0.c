static int cuvid_test_dummy_decoder(AVCodecContext *avctx,

                                    const CUVIDPARSERPARAMS *cuparseinfo,

                                    int probed_width,

                                    int probed_height)

{

    CuvidContext *ctx = avctx->priv_data;

    CUVIDDECODECREATEINFO cuinfo;

    CUvideodecoder cudec = 0;

    int ret = 0;



    memset(&cuinfo, 0, sizeof(cuinfo));



    cuinfo.CodecType = cuparseinfo->CodecType;

    cuinfo.ChromaFormat = cudaVideoChromaFormat_420;

    cuinfo.OutputFormat = cudaVideoSurfaceFormat_NV12;



    cuinfo.ulWidth = probed_width;

    cuinfo.ulHeight = probed_height;

    cuinfo.ulTargetWidth = cuinfo.ulWidth;

    cuinfo.ulTargetHeight = cuinfo.ulHeight;



    cuinfo.target_rect.left = 0;

    cuinfo.target_rect.top = 0;

    cuinfo.target_rect.right = cuinfo.ulWidth;

    cuinfo.target_rect.bottom = cuinfo.ulHeight;



    cuinfo.ulNumDecodeSurfaces = ctx->nb_surfaces;

    cuinfo.ulNumOutputSurfaces = 1;

    cuinfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;

    cuinfo.bitDepthMinus8 = 0;



    cuinfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;



    ret = CHECK_CU(ctx->cvdl->cuvidCreateDecoder(&cudec, &cuinfo));

    if (ret < 0)

        return ret;



    ret = CHECK_CU(ctx->cvdl->cuvidDestroyDecoder(cudec));

    if (ret < 0)

        return ret;



    return 0;

}

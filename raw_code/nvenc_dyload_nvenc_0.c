static av_cold int nvenc_dyload_nvenc(AVCodecContext *avctx)

{

    PNVENCODEAPICREATEINSTANCE nvEncodeAPICreateInstance = 0;

    NVENCSTATUS nvstatus;



    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;



    if (!nvenc_check_cuda(avctx))

        return 0;



    if (dl_fn->nvenc_lib)

        return 1;



#if defined(_WIN32)

    if (sizeof(void*) == 8) {

        dl_fn->nvenc_lib = LoadLibrary(TEXT("nvEncodeAPI64.dll"));

    } else {

        dl_fn->nvenc_lib = LoadLibrary(TEXT("nvEncodeAPI.dll"));

    }

#else

    dl_fn->nvenc_lib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);

#endif



    if (!dl_fn->nvenc_lib) {

        av_log(avctx, AV_LOG_FATAL, "Failed loading the nvenc library\n");

        goto error;

    }



    nvEncodeAPICreateInstance = (PNVENCODEAPICREATEINSTANCE)LOAD_FUNC(dl_fn->nvenc_lib, "NvEncodeAPICreateInstance");



    if (!nvEncodeAPICreateInstance) {

        av_log(avctx, AV_LOG_FATAL, "Failed to load nvenc entrypoint\n");

        goto error;

    }



    dl_fn->nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;



    nvstatus = nvEncodeAPICreateInstance(&dl_fn->nvenc_funcs);



    if (nvstatus != NV_ENC_SUCCESS) {

        nvenc_print_error(avctx, nvstatus, "Failed to create nvenc instance");

        goto error;

    }



    av_log(avctx, AV_LOG_VERBOSE, "Nvenc initialized successfully\n");



    return 1;



error:

    if (dl_fn->nvenc_lib)

        DL_CLOSE_FUNC(dl_fn->nvenc_lib);



    dl_fn->nvenc_lib = NULL;



    return 0;

}

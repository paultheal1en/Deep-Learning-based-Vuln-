static av_cold int nvenc_dyload_cuda(AVCodecContext *avctx)

{

    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;



#if CONFIG_CUDA

    dl_fn->cu_init                      = cuInit;

    dl_fn->cu_device_get_count          = cuDeviceGetCount;

    dl_fn->cu_device_get                = cuDeviceGet;

    dl_fn->cu_device_get_name           = cuDeviceGetName;

    dl_fn->cu_device_compute_capability = cuDeviceComputeCapability;

    dl_fn->cu_ctx_create                = cuCtxCreate_v2;

    dl_fn->cu_ctx_pop_current           = cuCtxPopCurrent_v2;

    dl_fn->cu_ctx_destroy               = cuCtxDestroy_v2;



    return 1;

#else

    if (dl_fn->cuda_lib)

        return 1;



#if defined(_WIN32)

    dl_fn->cuda_lib = LoadLibrary(TEXT("nvcuda.dll"));

#else

    dl_fn->cuda_lib = dlopen("libcuda.so", RTLD_LAZY);

#endif



    if (!dl_fn->cuda_lib) {

        av_log(avctx, AV_LOG_FATAL, "Failed loading CUDA library\n");

        goto error;

    }



    CHECK_LOAD_FUNC(PCUINIT, dl_fn->cu_init, "cuInit");

    CHECK_LOAD_FUNC(PCUDEVICEGETCOUNT, dl_fn->cu_device_get_count, "cuDeviceGetCount");

    CHECK_LOAD_FUNC(PCUDEVICEGET, dl_fn->cu_device_get, "cuDeviceGet");

    CHECK_LOAD_FUNC(PCUDEVICEGETNAME, dl_fn->cu_device_get_name, "cuDeviceGetName");

    CHECK_LOAD_FUNC(PCUDEVICECOMPUTECAPABILITY, dl_fn->cu_device_compute_capability, "cuDeviceComputeCapability");

    CHECK_LOAD_FUNC(PCUCTXCREATE, dl_fn->cu_ctx_create, "cuCtxCreate_v2");

    CHECK_LOAD_FUNC(PCUCTXPOPCURRENT, dl_fn->cu_ctx_pop_current, "cuCtxPopCurrent_v2");

    CHECK_LOAD_FUNC(PCUCTXDESTROY, dl_fn->cu_ctx_destroy, "cuCtxDestroy_v2");



    return 1;



error:



    if (dl_fn->cuda_lib)

        DL_CLOSE_FUNC(dl_fn->cuda_lib);



    dl_fn->cuda_lib = NULL;



    return 0;

#endif

}

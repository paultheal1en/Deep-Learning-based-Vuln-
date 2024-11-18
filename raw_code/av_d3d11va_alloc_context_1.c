AVD3D11VAContext *av_d3d11va_alloc_context(void)

{

    AVD3D11VAContext* res = av_mallocz(sizeof(AVD3D11VAContext));



    res->context_mutex = INVALID_HANDLE_VALUE;

    return res;

}
static int dxva2_map_frame(AVHWFramesContext *ctx, AVFrame *dst, const AVFrame *src,

                           int flags)

{

    IDirect3DSurface9 *surface = (IDirect3DSurface9*)src->data[3];

    DXVA2Mapping      *map;

    D3DSURFACE_DESC    surfaceDesc;

    D3DLOCKED_RECT     LockedRect;

    HRESULT            hr;

    int i, err, nb_planes;

    int lock_flags = 0;



    nb_planes = av_pix_fmt_count_planes(dst->format);



    hr = IDirect3DSurface9_GetDesc(surface, &surfaceDesc);

    if (FAILED(hr)) {

        av_log(ctx, AV_LOG_ERROR, "Error getting a surface description\n");

        return AVERROR_UNKNOWN;

    }



    if (!(flags & AV_HWFRAME_MAP_WRITE))

        lock_flags |= D3DLOCK_READONLY;

    if (flags & AV_HWFRAME_MAP_OVERWRITE)

        lock_flags |= D3DLOCK_DISCARD;



    hr = IDirect3DSurface9_LockRect(surface, &LockedRect, NULL, lock_flags);

    if (FAILED(hr)) {

        av_log(ctx, AV_LOG_ERROR, "Unable to lock DXVA2 surface\n");

        return AVERROR_UNKNOWN;

    }



    map = av_mallocz(sizeof(*map));

    if (!map)

        goto fail;



    err = ff_hwframe_map_create(src->hw_frames_ctx, dst, src,

                                dxva2_unmap_frame, map);

    if (err < 0) {

        av_freep(&map);

        goto fail;

    }



    for (i = 0; i < nb_planes; i++)

        dst->linesize[i] = LockedRect.Pitch;



    av_image_fill_pointers(dst->data, dst->format, surfaceDesc.Height,

                           (uint8_t*)LockedRect.pBits, dst->linesize);



    if (dst->format == AV_PIX_FMT_PAL8)

        dst->data[1] = (uint8_t*)map->palette_dummy;



    return 0;

fail:

    IDirect3DSurface9_UnlockRect(surface);

    return err;

}

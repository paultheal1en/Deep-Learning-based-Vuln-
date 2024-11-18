void ff_ivi_output_plane(IVIPlaneDesc *plane, uint8_t *dst, int dst_pitch)
{
    int             x, y;
    const int16_t   *src  = plane->bands[0].buf;
    uint32_t        pitch = plane->bands[0].pitch;
    for (y = 0; y < plane->height; y++) {
        for (x = 0; x < plane->width; x++)
            dst[x] = av_clip_uint8(src[x] + 128);
        src += pitch;
        dst += dst_pitch;
    }
}
static inline int is_yuv_planar(const PixFmtInfo *ps)

{

    return (ps->color_type == FF_COLOR_YUV ||

            ps->color_type == FF_COLOR_YUV_JPEG) &&

        ps->pixel_type == FF_PIXEL_PLANAR;

}

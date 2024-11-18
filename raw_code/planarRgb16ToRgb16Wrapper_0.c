static int planarRgb16ToRgb16Wrapper(SwsContext *c, const uint8_t *src[],

                                     int srcStride[], int srcSliceY, int srcSliceH,

                                     uint8_t *dst[], int dstStride[])

{

    const uint16_t *src102[] = { (uint16_t *)src[1], (uint16_t *)src[0], (uint16_t *)src[2] };

    const uint16_t *src201[] = { (uint16_t *)src[2], (uint16_t *)src[0], (uint16_t *)src[1] };

    int stride102[] = { srcStride[1], srcStride[0], srcStride[2] };

    int stride201[] = { srcStride[2], srcStride[0], srcStride[1] };

    const AVPixFmtDescriptor *src_format = av_pix_fmt_desc_get(c->srcFormat);

    const AVPixFmtDescriptor *dst_format = av_pix_fmt_desc_get(c->dstFormat);

    int bits_per_sample = src_format->comp[0].depth_minus1 + 1;

    int swap = 0;

    if ( HAVE_BIGENDIAN && !(src_format->flags & AV_PIX_FMT_FLAG_BE) ||

        !HAVE_BIGENDIAN &&   src_format->flags & AV_PIX_FMT_FLAG_BE)

        swap++;

    if ( HAVE_BIGENDIAN && !(dst_format->flags & AV_PIX_FMT_FLAG_BE) ||

        !HAVE_BIGENDIAN &&   dst_format->flags & AV_PIX_FMT_FLAG_BE)

        swap += 2;



    if ((src_format->flags & (AV_PIX_FMT_FLAG_PLANAR | AV_PIX_FMT_FLAG_RGB)) !=

        (AV_PIX_FMT_FLAG_PLANAR | AV_PIX_FMT_FLAG_RGB)) {

        av_log(c, AV_LOG_ERROR, "unsupported planar RGB conversion %s -> %s\n",

               src_format->name, dst_format->name);

        return srcSliceH;

    }

    switch (c->dstFormat) {

    case AV_PIX_FMT_BGR48LE:

    case AV_PIX_FMT_BGR48BE:

        gbr16ptopacked16(src102, stride102,

                         dst[0] + srcSliceY * dstStride[0], dstStride[0],

                         srcSliceH, 0, swap, bits_per_sample, c->srcW);

        break;

    case AV_PIX_FMT_RGB48LE:

    case AV_PIX_FMT_RGB48BE:

        gbr16ptopacked16(src201, stride201,

                         dst[0] + srcSliceY * dstStride[0], dstStride[0],

                         srcSliceH, 0, swap, bits_per_sample, c->srcW);

        break;

    case AV_PIX_FMT_RGBA64LE:

    case AV_PIX_FMT_RGBA64BE:

         gbr16ptopacked16(src201, stride201,

                          dst[0] + srcSliceY * dstStride[0], dstStride[0],

                          srcSliceH, 1, swap, bits_per_sample, c->srcW);

        break;

    case AV_PIX_FMT_BGRA64LE:

    case AV_PIX_FMT_BGRA64BE:

        gbr16ptopacked16(src102, stride102,

                         dst[0] + srcSliceY * dstStride[0], dstStride[0],

                         srcSliceH, 1, swap, bits_per_sample, c->srcW);

        break;

    default:

        av_log(c, AV_LOG_ERROR,

               "unsupported planar RGB conversion %s -> %s\n",

               src_format->name, dst_format->name);

    }



    return srcSliceH;

}

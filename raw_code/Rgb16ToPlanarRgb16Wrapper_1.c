static int Rgb16ToPlanarRgb16Wrapper(SwsContext *c, const uint8_t *src[],

                                     int srcStride[], int srcSliceY, int srcSliceH,

                                     uint8_t *dst[], int dstStride[])

{

    uint16_t *dst2013[] = { (uint16_t *)dst[2], (uint16_t *)dst[0], (uint16_t *)dst[1], (uint16_t *)dst[3] };

    uint16_t *dst1023[] = { (uint16_t *)dst[1], (uint16_t *)dst[0], (uint16_t *)dst[2], (uint16_t *)dst[3] };

    int stride2013[] = { dstStride[2], dstStride[0], dstStride[1], dstStride[3] };

    int stride1023[] = { dstStride[1], dstStride[0], dstStride[2], dstStride[3] };

    const AVPixFmtDescriptor *src_format = av_pix_fmt_desc_get(c->srcFormat);

    const AVPixFmtDescriptor *dst_format = av_pix_fmt_desc_get(c->dstFormat);

    int bpc = dst_format->comp[0].depth;

    int alpha = src_format->flags & AV_PIX_FMT_FLAG_ALPHA;

    int swap = 0;

    if ( HAVE_BIGENDIAN && !(src_format->flags & AV_PIX_FMT_FLAG_BE) ||

        !HAVE_BIGENDIAN &&   src_format->flags & AV_PIX_FMT_FLAG_BE)

        swap++;

    if ( HAVE_BIGENDIAN && !(dst_format->flags & AV_PIX_FMT_FLAG_BE) ||

        !HAVE_BIGENDIAN &&   dst_format->flags & AV_PIX_FMT_FLAG_BE)

        swap += 2;



    if ((dst_format->flags & (AV_PIX_FMT_FLAG_PLANAR | AV_PIX_FMT_FLAG_RGB)) !=

        (AV_PIX_FMT_FLAG_PLANAR | AV_PIX_FMT_FLAG_RGB) || bpc < 9) {

        av_log(c, AV_LOG_ERROR, "unsupported conversion to planar RGB %s -> %s\n",

               src_format->name, dst_format->name);

        return srcSliceH;

    }

    switch (c->srcFormat) {

    case AV_PIX_FMT_RGB48LE:

    case AV_PIX_FMT_RGB48BE:

    case AV_PIX_FMT_RGBA64LE:

    case AV_PIX_FMT_RGBA64BE:

        packed16togbra16(src[0] + srcSliceY * srcStride[0], srcStride[0],

                         dst2013, stride2013, srcSliceH, alpha, swap,

                         16 - bpc, c->srcW);

        break;

    case AV_PIX_FMT_BGR48LE:

    case AV_PIX_FMT_BGR48BE:

    case AV_PIX_FMT_BGRA64LE:

    case AV_PIX_FMT_BGRA64BE:

        packed16togbra16(src[0] + srcSliceY * srcStride[0], srcStride[0],

                         dst1023, stride1023, srcSliceH, alpha, swap,

                         16 - bpc, c->srcW);

        break;

    default:

        av_log(c, AV_LOG_ERROR,

               "unsupported conversion to planar RGB %s -> %s\n",

               src_format->name, dst_format->name);

    }



    return srcSliceH;

}

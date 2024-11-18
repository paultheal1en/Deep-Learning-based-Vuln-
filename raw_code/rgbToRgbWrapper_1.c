static int rgbToRgbWrapper(SwsContext *c, const uint8_t *src[], int srcStride[],

                           int srcSliceY, int srcSliceH, uint8_t *dst[],

                           int dstStride[])

{

    const enum PixelFormat srcFormat = c->srcFormat;

    const enum PixelFormat dstFormat = c->dstFormat;

    const int srcBpp = (c->srcFormatBpp + 7) >> 3;

    const int dstBpp = (c->dstFormatBpp + 7) >> 3;

    const int srcId = c->srcFormatBpp;

    const int dstId = c->dstFormatBpp;

    void (*conv)(const uint8_t *src, uint8_t *dst, int src_size) = NULL;



#define CONV_IS(src, dst) (srcFormat == PIX_FMT_##src && dstFormat == PIX_FMT_##dst)



    if (isRGBA32(srcFormat) && isRGBA32(dstFormat)) {

        if (     CONV_IS(ABGR, RGBA)

              || CONV_IS(ARGB, BGRA)

              || CONV_IS(BGRA, ARGB)

              || CONV_IS(RGBA, ABGR)) conv = shuffle_bytes_3210;

        else if (CONV_IS(ABGR, ARGB)

              || CONV_IS(ARGB, ABGR)) conv = shuffle_bytes_0321;

        else if (CONV_IS(ABGR, BGRA)

              || CONV_IS(ARGB, RGBA)) conv = shuffle_bytes_1230;

        else if (CONV_IS(BGRA, RGBA)

              || CONV_IS(RGBA, BGRA)) conv = shuffle_bytes_2103;

        else if (CONV_IS(BGRA, ABGR)

              || CONV_IS(RGBA, ARGB)) conv = shuffle_bytes_3012;

    } else

    /* BGR -> BGR */

    if ((isBGRinInt(srcFormat) && isBGRinInt(dstFormat)) ||

        (isRGBinInt(srcFormat) && isRGBinInt(dstFormat))) {

        switch (srcId | (dstId << 16)) {

        case 0x000F0010: conv = rgb16to15; break;

        case 0x000F0018: conv = rgb24to15; break;

        case 0x000F0020: conv = rgb32to15; break;

        case 0x0010000F: conv = rgb15to16; break;

        case 0x00100018: conv = rgb24to16; break;

        case 0x00100020: conv = rgb32to16; break;

        case 0x0018000F: conv = rgb15to24; break;

        case 0x00180010: conv = rgb16to24; break;

        case 0x00180020: conv = rgb32to24; break;

        case 0x0020000F: conv = rgb15to32; break;

        case 0x00200010: conv = rgb16to32; break;

        case 0x00200018: conv = rgb24to32; break;

        }

    } else if ((isBGRinInt(srcFormat) && isRGBinInt(dstFormat)) ||

               (isRGBinInt(srcFormat) && isBGRinInt(dstFormat))) {

        switch (srcId | (dstId << 16)) {

        case 0x000C000C: conv = rgb12tobgr12; break;

        case 0x000F000F: conv = rgb15tobgr15; break;

        case 0x000F0010: conv = rgb16tobgr15; break;

        case 0x000F0018: conv = rgb24tobgr15; break;

        case 0x000F0020: conv = rgb32tobgr15; break;

        case 0x0010000F: conv = rgb15tobgr16; break;

        case 0x00100010: conv = rgb16tobgr16; break;

        case 0x00100018: conv = rgb24tobgr16; break;

        case 0x00100020: conv = rgb32tobgr16; break;

        case 0x0018000F: conv = rgb15tobgr24; break;

        case 0x00180010: conv = rgb16tobgr24; break;

        case 0x00180018: conv = rgb24tobgr24; break;

        case 0x00180020: conv = rgb32tobgr24; break;

        case 0x0020000F: conv = rgb15tobgr32; break;

        case 0x00200010: conv = rgb16tobgr32; break;

        case 0x00200018: conv = rgb24tobgr32; break;

        }

    }



    if (!conv) {

        av_log(c, AV_LOG_ERROR, "internal error %s -> %s converter\n",

               sws_format_name(srcFormat), sws_format_name(dstFormat));

    } else {

        const uint8_t *srcPtr = src[0];

              uint8_t *dstPtr = dst[0];

        if ((srcFormat == PIX_FMT_RGB32_1 || srcFormat == PIX_FMT_BGR32_1) &&

            !isRGBA32(dstFormat))

            srcPtr += ALT32_CORR;



        if ((dstFormat == PIX_FMT_RGB32_1 || dstFormat == PIX_FMT_BGR32_1) &&

            !isRGBA32(srcFormat))

            dstPtr += ALT32_CORR;



        if (dstStride[0] * srcBpp == srcStride[0] * dstBpp && srcStride[0] > 0 &&

            !(srcStride[0] % srcBpp))

            conv(srcPtr, dstPtr + dstStride[0] * srcSliceY,

                 srcSliceH * srcStride[0]);

        else {

            int i;

            dstPtr += dstStride[0] * srcSliceY;



            for (i = 0; i < srcSliceH; i++) {

                conv(srcPtr, dstPtr, c->srcW * srcBpp);

                srcPtr += srcStride[0];

                dstPtr += dstStride[0];

            }

        }

    }

    return srcSliceH;

}

void ff_init_vscale_pfn(SwsContext *c,

    yuv2planar1_fn yuv2plane1,

    yuv2planarX_fn yuv2planeX,

    yuv2interleavedX_fn yuv2nv12cX,

    yuv2packed1_fn yuv2packed1,

    yuv2packed2_fn yuv2packed2,

    yuv2packedX_fn yuv2packedX,

    yuv2anyX_fn yuv2anyX, int use_mmx)

{

    VScalerContext *lumCtx = NULL;

    VScalerContext *chrCtx = NULL;

    int idx = c->numDesc - (c->is_internal_gamma ? 2 : 1); //FIXME avoid hardcoding indexes



    if (isPlanarYUV(c->dstFormat) || (isGray(c->dstFormat) && !isALPHA(c->dstFormat))) {

        if (!isGray(c->dstFormat)) {

            chrCtx = c->desc[idx].instance;



            chrCtx->filter[0] = use_mmx ? (int16_t*)c->chrMmxFilter : c->vChrFilter;

            chrCtx->filter_size = c->vChrFilterSize;

            chrCtx->filter_pos = c->vChrFilterPos;

            chrCtx->isMMX = use_mmx;



            --idx;

            if (yuv2nv12cX)               chrCtx->pfn = yuv2nv12cX;

            else if (c->vChrFilterSize == 1) chrCtx->pfn = yuv2plane1;

            else                             chrCtx->pfn = yuv2planeX;

        }



        lumCtx = c->desc[idx].instance;



        lumCtx->filter[0] = use_mmx ? (int16_t*)c->lumMmxFilter : c->vLumFilter;

        lumCtx->filter[1] = use_mmx ? (int16_t*)c->alpMmxFilter : c->vLumFilter;

        lumCtx->filter_size = c->vLumFilterSize;

        lumCtx->filter_pos = c->vLumFilterPos;

        lumCtx->isMMX = use_mmx;



        if (c->vLumFilterSize == 1) lumCtx->pfn = yuv2plane1;

        else                        lumCtx->pfn = yuv2planeX;



    } else {

        lumCtx = c->desc[idx].instance;

        chrCtx = &lumCtx[1];



        lumCtx->filter[0] = c->vLumFilter;

        lumCtx->filter_size = c->vLumFilterSize;

        lumCtx->filter_pos = c->vLumFilterPos;



        chrCtx->filter[0] = c->vChrFilter;

        chrCtx->filter_size = c->vChrFilterSize;

        chrCtx->filter_pos = c->vChrFilterPos;



        lumCtx->isMMX = use_mmx;

        chrCtx->isMMX = use_mmx;



        if (yuv2packedX) {

            if (c->yuv2packed1 && c->vLumFilterSize == 1 && c->vChrFilterSize <= 2)

                lumCtx->pfn = yuv2packed1;

            else if (c->yuv2packed2 && c->vLumFilterSize == 2 && c->vChrFilterSize == 2)

                lumCtx->pfn = yuv2packed2;

            else

                lumCtx->pfn = yuv2packedX;

        } else

            lumCtx->pfn = yuv2anyX;

    }

}

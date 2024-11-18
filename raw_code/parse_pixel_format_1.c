static int parse_pixel_format(AVCodecContext *avctx)
{
    DDSContext *ctx = avctx->priv_data;
    GetByteContext *gbc = &ctx->gbc;
    char buf[32];
    uint32_t flags, fourcc, gimp_tag;
    enum DDSDXGIFormat dxgi;
    int size, bpp, r, g, b, a;
    int alpha_exponent, ycocg_classic, ycocg_scaled, normal_map, array;
    /* Alternative DDS implementations use reserved1 as custom header. */
    bytestream2_skip(gbc, 4 * 3);
    gimp_tag = bytestream2_get_le32(gbc);
    alpha_exponent = gimp_tag == MKTAG('A', 'E', 'X', 'P');
    ycocg_classic  = gimp_tag == MKTAG('Y', 'C', 'G', '1');
    ycocg_scaled   = gimp_tag == MKTAG('Y', 'C', 'G', '2');
    bytestream2_skip(gbc, 4 * 7);
    /* Now the real DDPF starts. */
    size = bytestream2_get_le32(gbc);
    if (size != 32) {
        av_log(avctx, AV_LOG_ERROR, "Invalid pixel format header %d.\n", size);
        return AVERROR_INVALIDDATA;
    flags = bytestream2_get_le32(gbc);
    ctx->compressed = flags & DDPF_FOURCC;
    ctx->paletted   = flags & DDPF_PALETTE;
    normal_map      = flags & DDPF_NORMALMAP;
    fourcc = bytestream2_get_le32(gbc);
    bpp = bytestream2_get_le32(gbc); // rgbbitcount
    r   = bytestream2_get_le32(gbc); // rbitmask
    g   = bytestream2_get_le32(gbc); // gbitmask
    b   = bytestream2_get_le32(gbc); // bbitmask
    a   = bytestream2_get_le32(gbc); // abitmask
    bytestream2_skip(gbc, 4); // caps
    bytestream2_skip(gbc, 4); // caps2
    bytestream2_skip(gbc, 4); // caps3
    bytestream2_skip(gbc, 4); // caps4
    bytestream2_skip(gbc, 4); // reserved2
    av_get_codec_tag_string(buf, sizeof(buf), fourcc);
    av_log(avctx, AV_LOG_VERBOSE, "fourcc %s bpp %d "
           "r 0x%x g 0x%x b 0x%x a 0x%x\n", buf, bpp, r, g, b, a);
    if (gimp_tag) {
        av_get_codec_tag_string(buf, sizeof(buf), gimp_tag);
        av_log(avctx, AV_LOG_VERBOSE, "and GIMP-DDS tag %s\n", buf);
    if (ctx->compressed)
        avctx->pix_fmt = AV_PIX_FMT_RGBA;
    if (ctx->compressed) {
        switch (fourcc) {
        case MKTAG('D', 'X', 'T', '1'):
            ctx->tex_ratio = 8;
            ctx->tex_funct = ctx->texdsp.dxt1a_block;
            break;
        case MKTAG('D', 'X', 'T', '2'):
            ctx->tex_ratio = 16;
            ctx->tex_funct = ctx->texdsp.dxt2_block;
            break;
        case MKTAG('D', 'X', 'T', '3'):
            ctx->tex_ratio = 16;
            ctx->tex_funct = ctx->texdsp.dxt3_block;
            break;
        case MKTAG('D', 'X', 'T', '4'):
            ctx->tex_ratio = 16;
            ctx->tex_funct = ctx->texdsp.dxt4_block;
            break;
        case MKTAG('D', 'X', 'T', '5'):
            ctx->tex_ratio = 16;
            if (ycocg_scaled)
                ctx->tex_funct = ctx->texdsp.dxt5ys_block;
            else if (ycocg_classic)
                ctx->tex_funct = ctx->texdsp.dxt5y_block;
            else
                ctx->tex_funct = ctx->texdsp.dxt5_block;
            break;
        case MKTAG('R', 'X', 'G', 'B'):
            ctx->tex_ratio = 16;
            ctx->tex_funct = ctx->texdsp.dxt5_block;
            /* This format may be considered as a normal map,
             * but it is handled differently in a separate postproc. */
            ctx->postproc = DDS_SWIZZLE_RXGB;
            normal_map = 0;
            break;
        case MKTAG('A', 'T', 'I', '1'):
        case MKTAG('B', 'C', '4', 'U'):
            ctx->tex_ratio = 8;
            ctx->tex_funct = ctx->texdsp.rgtc1u_block;
            break;
        case MKTAG('B', 'C', '4', 'S'):
            ctx->tex_ratio = 8;
            ctx->tex_funct = ctx->texdsp.rgtc1s_block;
            break;
        case MKTAG('A', 'T', 'I', '2'):
            /* RGT2 variant with swapped R and G (3Dc)*/
            ctx->tex_ratio = 16;
            ctx->tex_funct = ctx->texdsp.dxn3dc_block;
            break;
        case MKTAG('B', 'C', '5', 'U'):
            ctx->tex_ratio = 16;
            ctx->tex_funct = ctx->texdsp.rgtc2u_block;
            break;
        case MKTAG('B', 'C', '5', 'S'):
            ctx->tex_ratio = 16;
            ctx->tex_funct = ctx->texdsp.rgtc2s_block;
            break;
        case MKTAG('U', 'Y', 'V', 'Y'):
            ctx->compressed = 0;
            avctx->pix_fmt = AV_PIX_FMT_UYVY422;
            break;
        case MKTAG('Y', 'U', 'Y', '2'):
            ctx->compressed = 0;
            avctx->pix_fmt = AV_PIX_FMT_YUYV422;
            break;
        case MKTAG('P', '8', ' ', ' '):
            /* ATI Palette8, same as normal palette */
            ctx->compressed = 0;
            ctx->paletted   = 1;
            avctx->pix_fmt  = AV_PIX_FMT_PAL8;
            break;
        case MKTAG('D', 'X', '1', '0'):
            /* DirectX 10 extra header */
            dxgi = bytestream2_get_le32(gbc);
            bytestream2_skip(gbc, 4); // resourceDimension
            bytestream2_skip(gbc, 4); // miscFlag
            array = bytestream2_get_le32(gbc);
            bytestream2_skip(gbc, 4); // miscFlag2
            if (array != 0)
                av_log(avctx, AV_LOG_VERBOSE,
                       "Found array of size %d (ignored).\n", array);
            /* Only BC[1-5] are actually compressed. */
            ctx->compressed = (dxgi >= 70) && (dxgi <= 84);
            av_log(avctx, AV_LOG_VERBOSE, "DXGI format %d.\n", dxgi);
            switch (dxgi) {
            /* RGB types. */
            case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
            case DXGI_FORMAT_R16G16B16A16_UNORM:
            case DXGI_FORMAT_R16G16B16A16_UINT:
            case DXGI_FORMAT_R16G16B16A16_SNORM:
            case DXGI_FORMAT_R16G16B16A16_SINT:
                avctx->pix_fmt = AV_PIX_FMT_BGRA64;
                break;
            case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
                avctx->colorspace = AVCOL_SPC_RGB;
            case DXGI_FORMAT_R8G8B8A8_TYPELESS:
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            case DXGI_FORMAT_R8G8B8A8_UINT:
            case DXGI_FORMAT_R8G8B8A8_SNORM:
            case DXGI_FORMAT_R8G8B8A8_SINT:
                avctx->pix_fmt = AV_PIX_FMT_BGRA;
                break;
            case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
                avctx->colorspace = AVCOL_SPC_RGB;
            case DXGI_FORMAT_B8G8R8A8_TYPELESS:
            case DXGI_FORMAT_B8G8R8A8_UNORM:
                avctx->pix_fmt = AV_PIX_FMT_RGBA;
                break;
            case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
                avctx->colorspace = AVCOL_SPC_RGB;
            case DXGI_FORMAT_B8G8R8X8_TYPELESS:
            case DXGI_FORMAT_B8G8R8X8_UNORM:
                avctx->pix_fmt = AV_PIX_FMT_RGBA; // opaque
                break;
            case DXGI_FORMAT_B5G6R5_UNORM:
                avctx->pix_fmt = AV_PIX_FMT_RGB565LE;
                break;
            /* Texture types. */
            case DXGI_FORMAT_BC1_UNORM_SRGB:
                avctx->colorspace = AVCOL_SPC_RGB;
            case DXGI_FORMAT_BC1_TYPELESS:
            case DXGI_FORMAT_BC1_UNORM:
                ctx->tex_ratio = 8;
                ctx->tex_funct = ctx->texdsp.dxt1a_block;
                break;
            case DXGI_FORMAT_BC2_UNORM_SRGB:
                avctx->colorspace = AVCOL_SPC_RGB;
            case DXGI_FORMAT_BC2_TYPELESS:
            case DXGI_FORMAT_BC2_UNORM:
                ctx->tex_ratio = 16;
                ctx->tex_funct = ctx->texdsp.dxt3_block;
                break;
            case DXGI_FORMAT_BC3_UNORM_SRGB:
                avctx->colorspace = AVCOL_SPC_RGB;
            case DXGI_FORMAT_BC3_TYPELESS:
            case DXGI_FORMAT_BC3_UNORM:
                ctx->tex_ratio = 16;
                ctx->tex_funct = ctx->texdsp.dxt5_block;
                break;
            case DXGI_FORMAT_BC4_TYPELESS:
            case DXGI_FORMAT_BC4_UNORM:
                ctx->tex_ratio = 8;
                ctx->tex_funct = ctx->texdsp.rgtc1u_block;
                break;
            case DXGI_FORMAT_BC4_SNORM:
                ctx->tex_ratio = 8;
                ctx->tex_funct = ctx->texdsp.rgtc1s_block;
                break;
            case DXGI_FORMAT_BC5_TYPELESS:
            case DXGI_FORMAT_BC5_UNORM:
                ctx->tex_ratio = 16;
                ctx->tex_funct = ctx->texdsp.rgtc2u_block;
                break;
            case DXGI_FORMAT_BC5_SNORM:
                ctx->tex_ratio = 16;
                ctx->tex_funct = ctx->texdsp.rgtc2s_block;
                break;
            default:
                av_log(avctx, AV_LOG_ERROR,
                       "Unsupported DXGI format %d.\n", dxgi);
                return AVERROR_INVALIDDATA;
            break;
        default:
            av_log(avctx, AV_LOG_ERROR, "Unsupported %s fourcc.\n", buf);
            return AVERROR_INVALIDDATA;
    } else if (ctx->paletted) {
        if (bpp == 8) {
            avctx->pix_fmt = AV_PIX_FMT_PAL8;
        } else {
            av_log(avctx, AV_LOG_ERROR, "Unsupported palette bpp %d.\n", bpp);
            return AVERROR_INVALIDDATA;
    } else {
        /*  8 bpp */
        if (bpp == 8 && r == 0xff && g == 0 && b == 0 && a == 0)
            avctx->pix_fmt = AV_PIX_FMT_GRAY8;
        /* 16 bpp */
        else if (bpp == 16 && r == 0xff && g == 0 && b == 0 && a == 0xff00)
            avctx->pix_fmt = AV_PIX_FMT_YA8;
        else if (bpp == 16 && r == 0xffff && g == 0 && b == 0 && a == 0)
            avctx->pix_fmt = AV_PIX_FMT_GRAY16LE;
        else if (bpp == 16 && r == 0xf800 && g == 0x7e0 && b == 0x1f && a == 0)
            avctx->pix_fmt = AV_PIX_FMT_RGB565LE;
        /* 24 bpp */
        else if (bpp == 24 && r == 0xff0000 && g == 0xff00 && b == 0xff && a == 0)
            avctx->pix_fmt = AV_PIX_FMT_BGR24;
        /* 32 bpp */
        else if (bpp == 32 && r == 0xff0000 && g == 0xff00 && b == 0xff && a == 0)
            avctx->pix_fmt = AV_PIX_FMT_BGRA; // opaque
        else if (bpp == 32 && r == 0xff && g == 0xff00 && b == 0xff0000 && a == 0)
            avctx->pix_fmt = AV_PIX_FMT_RGBA; // opaque
        else if (bpp == 32 && r == 0xff0000 && g == 0xff00 && b == 0xff && a == 0xff000000)
            avctx->pix_fmt = AV_PIX_FMT_BGRA;
        else if (bpp == 32 && r == 0xff && g == 0xff00 && b == 0xff0000 && a == 0xff000000)
            avctx->pix_fmt = AV_PIX_FMT_RGBA;
        /* give up */
        else {
            av_log(avctx, AV_LOG_ERROR, "Unknown pixel format "
                   "[bpp %d r 0x%x g 0x%x b 0x%x a 0x%x].\n", bpp, r, g, b, a);
            return AVERROR_INVALIDDATA;
    /* Set any remaining post-proc that should happen before frame is ready. */
    if (alpha_exponent)
        ctx->postproc = DDS_ALPHA_EXP;
    else if (normal_map)
        ctx->postproc = DDS_NORMAL_MAP;
    else if (ycocg_classic && !ctx->compressed)
        ctx->postproc = DDS_RAW_YCOCG;
    else if (avctx->pix_fmt == AV_PIX_FMT_YA8)
        ctx->postproc = DDS_SWAP_ALPHA;
    /* ATI/NVidia variants sometimes add swizzling in bpp. */
    switch (bpp) {
    case MKTAG('A', '2', 'X', 'Y'):
        ctx->postproc = DDS_SWIZZLE_A2XY;
        break;
    case MKTAG('x', 'G', 'B', 'R'):
        ctx->postproc = DDS_SWIZZLE_XGBR;
        break;
    case MKTAG('x', 'R', 'B', 'G'):
        ctx->postproc = DDS_SWIZZLE_XRBG;
        break;
    case MKTAG('R', 'B', 'x', 'G'):
        ctx->postproc = DDS_SWIZZLE_RBXG;
        break;
    case MKTAG('R', 'G', 'x', 'B'):
        ctx->postproc = DDS_SWIZZLE_RGXB;
        break;
    case MKTAG('R', 'x', 'B', 'G'):
        ctx->postproc = DDS_SWIZZLE_RXBG;
        break;
    case MKTAG('x', 'G', 'x', 'R'):
        ctx->postproc = DDS_SWIZZLE_XGXR;
        break;
    case MKTAG('A', '2', 'D', '5'):
        ctx->postproc = DDS_NORMAL_MAP;
        break;
    return 0;
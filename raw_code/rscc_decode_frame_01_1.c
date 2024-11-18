static int rscc_decode_frame(AVCodecContext *avctx, void *data,
                                     int *got_frame, AVPacket *avpkt)
{
    RsccContext *ctx = avctx->priv_data;
    GetByteContext *gbc = &ctx->gbc;
    GetByteContext tiles_gbc;
    AVFrame *frame = data;
    const uint8_t *pixels, *raw;
    uint8_t *inflated_tiles = NULL;
    int tiles_nb, packed_size, pixel_size = 0;
    int i, ret = 0;
    bytestream2_init(gbc, avpkt->data, avpkt->size);
    /* Size check */
    if (bytestream2_get_bytes_left(gbc) < 12) {
        av_log(avctx, AV_LOG_ERROR, "Packet too small (%d)\n", avpkt->size);
        return AVERROR_INVALIDDATA;
    /* Read number of tiles, and allocate the array */
    tiles_nb = bytestream2_get_le16(gbc);
    av_fast_malloc(&ctx->tiles, &ctx->tiles_size,
                   tiles_nb * sizeof(*ctx->tiles));
    if (!ctx->tiles) {
        ret = AVERROR(ENOMEM);
    av_log(avctx, AV_LOG_DEBUG, "Frame with %d tiles.\n", tiles_nb);
    /* When there are more than 5 tiles, they are packed together with
     * a size header. When that size does not match the number of tiles
     * times the tile size, it means it needs to be inflated as well */
    if (tiles_nb > 5) {
        uLongf packed_tiles_size;
        if (tiles_nb < 32)
            packed_tiles_size = bytestream2_get_byte(gbc);
        else
            packed_tiles_size = bytestream2_get_le16(gbc);
        ff_dlog(avctx, "packed tiles of size %lu.\n", packed_tiles_size);
        /* If necessary, uncompress tiles, and hijack the bytestream reader */
        if (packed_tiles_size != tiles_nb * TILE_SIZE) {
            uLongf length = tiles_nb * TILE_SIZE;
            inflated_tiles = av_malloc(length);
            if (!inflated_tiles) {
                ret = AVERROR(ENOMEM);
            ret = uncompress(inflated_tiles, &length,
                             gbc->buffer, packed_tiles_size);
            if (ret) {
                av_log(avctx, AV_LOG_ERROR, "Tile deflate error %d.\n", ret);
                ret = AVERROR_UNKNOWN;
            /* Skip the compressed tile section in the main byte reader,
             * and point it to read the newly uncompressed data */
            bytestream2_skip(gbc, packed_tiles_size);
            bytestream2_init(&tiles_gbc, inflated_tiles, length);
            gbc = &tiles_gbc;
    /* Fill in array of tiles, keeping track of how many pixels are updated */
    for (i = 0; i < tiles_nb; i++) {
        ctx->tiles[i].x = bytestream2_get_le16(gbc);
        ctx->tiles[i].w = bytestream2_get_le16(gbc);
        ctx->tiles[i].y = bytestream2_get_le16(gbc);
        ctx->tiles[i].h = bytestream2_get_le16(gbc);
        pixel_size += ctx->tiles[i].w * ctx->tiles[i].h * ctx->component_size;
        ff_dlog(avctx, "tile %d orig(%d,%d) %dx%d.\n", i,
                ctx->tiles[i].x, ctx->tiles[i].y,
                ctx->tiles[i].w, ctx->tiles[i].h);
        if (ctx->tiles[i].w == 0 || ctx->tiles[i].h == 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "invalid tile %d at (%d.%d) with size %dx%d.\n", i,
                   ctx->tiles[i].x, ctx->tiles[i].y,
                   ctx->tiles[i].w, ctx->tiles[i].h);
        } else if (ctx->tiles[i].x + ctx->tiles[i].w > avctx->width ||
                   ctx->tiles[i].y + ctx->tiles[i].h > avctx->height) {
            av_log(avctx, AV_LOG_ERROR,
                   "out of bounds tile %d at (%d.%d) with size %dx%d.\n", i,
                   ctx->tiles[i].x, ctx->tiles[i].y,
                   ctx->tiles[i].w, ctx->tiles[i].h);
    /* Reset the reader in case it had been modified before */
    gbc = &ctx->gbc;
    /* Extract how much pixel data the tiles contain */
    if (pixel_size < 0x100)
        packed_size = bytestream2_get_byte(gbc);
    else if (pixel_size < 0x10000)
        packed_size = bytestream2_get_le16(gbc);
    else if (pixel_size < 0x1000000)
        packed_size = bytestream2_get_le24(gbc);
    else
        packed_size = bytestream2_get_le32(gbc);
    ff_dlog(avctx, "pixel_size %d packed_size %d.\n", pixel_size, packed_size);
    if (packed_size < 0) {
        av_log(avctx, AV_LOG_ERROR, "Invalid tile size %d\n", packed_size);
    /* Get pixels buffer, it may be deflated or just raw */
    if (pixel_size == packed_size) {
        if (bytestream2_get_bytes_left(gbc) < pixel_size) {
            av_log(avctx, AV_LOG_ERROR, "Insufficient input for %d\n", pixel_size);
        pixels = gbc->buffer;
    } else {
        uLongf len = ctx->inflated_size;
        if (bytestream2_get_bytes_left(gbc) < packed_size) {
            av_log(avctx, AV_LOG_ERROR, "Insufficient input for %d\n", packed_size);
        ret = uncompress(ctx->inflated_buf, &len, gbc->buffer, packed_size);
        if (ret) {
            av_log(avctx, AV_LOG_ERROR, "Pixel deflate error %d.\n", ret);
            ret = AVERROR_UNKNOWN;
        pixels = ctx->inflated_buf;
    /* Allocate when needed */
    ret = ff_reget_buffer(avctx, ctx->reference);
    if (ret < 0)
    /* Pointer to actual pixels, will be updated when data is consumed */
    raw = pixels;
    for (i = 0; i < tiles_nb; i++) {
        uint8_t *dst = ctx->reference->data[0] + ctx->reference->linesize[0] *
                       (avctx->height - ctx->tiles[i].y - 1) +
                       ctx->tiles[i].x * ctx->component_size;
        av_image_copy_plane(dst, -1 * ctx->reference->linesize[0],
                            raw, ctx->tiles[i].w * ctx->component_size,
                            ctx->tiles[i].w * ctx->component_size,
                            ctx->tiles[i].h);
        raw += ctx->tiles[i].w * ctx->component_size * ctx->tiles[i].h;
    /* Frame is ready to be output */
    ret = av_frame_ref(frame, ctx->reference);
    if (ret < 0)
    /* Keyframe when the number of pixels updated matches the whole surface */
    if (pixel_size == ctx->inflated_size) {
        frame->pict_type = AV_PICTURE_TYPE_I;
        frame->key_frame = 1;
    } else {
        frame->pict_type = AV_PICTURE_TYPE_P;
    /* Palette handling */
    if (avctx->pix_fmt == AV_PIX_FMT_PAL8) {
        int size;
        const uint8_t *palette = av_packet_get_side_data(avpkt,
                                                         AV_PKT_DATA_PALETTE,
                                                         &size);
        if (palette && size == AVPALETTE_SIZE) {
            frame->palette_has_changed = 1;
            memcpy(ctx->palette, palette, AVPALETTE_SIZE);
        } else if (palette) {
            av_log(avctx, AV_LOG_ERROR, "Palette size %d is wrong\n", size);
        memcpy (frame->data[1], ctx->palette, AVPALETTE_SIZE);
    *got_frame = 1;
    ret = avpkt->size;
end:
    av_free(inflated_tiles);
    return ret;
static int escape124_decode_frame(AVCodecContext *avctx,
                                  void *data, int *got_frame,
                                  AVPacket *avpkt)
{
    int buf_size = avpkt->size;
    Escape124Context *s = avctx->priv_data;
    AVFrame *frame = data;
    GetBitContext gb;
    unsigned frame_flags, frame_size;
    unsigned i;
    unsigned superblock_index, cb_index = 1,
             superblock_col_index = 0,
             superblocks_per_row = avctx->width / 8, skip = -1;
    uint16_t* old_frame_data, *new_frame_data;
    unsigned old_stride, new_stride;
    int ret;
    if ((ret = init_get_bits8(&gb, avpkt->data, avpkt->size)) < 0)
        return ret;
    // This call also guards the potential depth reads for the
    // codebook unpacking.
    if (get_bits_left(&gb) < 64)
        return -1;
    frame_flags = get_bits_long(&gb, 32);
    frame_size  = get_bits_long(&gb, 32);
    // Leave last frame unchanged
    // FIXME: Is this necessary?  I haven't seen it in any real samples
    if (!(frame_flags & 0x114) || !(frame_flags & 0x7800000)) {
        if (!s->frame->data[0])
        av_log(avctx, AV_LOG_DEBUG, "Skipping frame\n");
        *got_frame = 1;
        if ((ret = av_frame_ref(frame, s->frame)) < 0)
            return ret;
        return frame_size;
    for (i = 0; i < 3; i++) {
        if (frame_flags & (1 << (17 + i))) {
            unsigned cb_depth, cb_size;
            if (i == 2) {
                // This codebook can be cut off at places other than
                // powers of 2, leaving some of the entries undefined.
                cb_size = get_bits_long(&gb, 20);
                if (!cb_size) {
                    av_log(avctx, AV_LOG_ERROR, "Invalid codebook size 0.\n");
                cb_depth = av_log2(cb_size - 1) + 1;
            } else {
                cb_depth = get_bits(&gb, 4);
                if (i == 0) {
                    // This is the most basic codebook: pow(2,depth) entries
                    // for a depth-length key
                    cb_size = 1 << cb_depth;
                } else {
                    // This codebook varies per superblock
                    // FIXME: I don't think this handles integer overflow
                    // properly
                    cb_size = s->num_superblocks << cb_depth;
            av_freep(&s->codebooks[i].blocks);
            s->codebooks[i] = unpack_codebook(&gb, cb_depth, cb_size);
            if (!s->codebooks[i].blocks)
                return -1;
    if ((ret = ff_get_buffer(avctx, frame, AV_GET_BUFFER_FLAG_REF)) < 0)
        return ret;
    new_frame_data = (uint16_t*)frame->data[0];
    new_stride = frame->linesize[0] / 2;
    old_frame_data = (uint16_t*)s->frame->data[0];
    old_stride = s->frame->linesize[0] / 2;
    for (superblock_index = 0; superblock_index < s->num_superblocks;
         superblock_index++) {
        MacroBlock mb;
        SuperBlock sb;
        unsigned multi_mask = 0;
        if (skip == -1) {
            // Note that this call will make us skip the rest of the blocks
            // if the frame prematurely ends
            skip = decode_skip_count(&gb);
        if (skip) {
            copy_superblock(new_frame_data, new_stride,
                            old_frame_data, old_stride);
        } else {
            copy_superblock(sb.pixels, 8,
                            old_frame_data, old_stride);
            while (get_bits_left(&gb) >= 1 && !get_bits1(&gb)) {
                unsigned mask;
                mb = decode_macroblock(s, &gb, &cb_index, superblock_index);
                mask = get_bits(&gb, 16);
                multi_mask |= mask;
                for (i = 0; i < 16; i++) {
                    if (mask & mask_matrix[i]) {
                        insert_mb_into_sb(&sb, mb, i);
            if (!get_bits1(&gb)) {
                unsigned inv_mask = get_bits(&gb, 4);
                for (i = 0; i < 4; i++) {
                    if (inv_mask & (1 << i)) {
                        multi_mask ^= 0xF << i*4;
                    } else {
                        multi_mask ^= get_bits(&gb, 4) << i*4;
                for (i = 0; i < 16; i++) {
                    if (multi_mask & mask_matrix[i]) {
                        mb = decode_macroblock(s, &gb, &cb_index,
                                               superblock_index);
                        insert_mb_into_sb(&sb, mb, i);
            } else if (frame_flags & (1 << 16)) {
                while (get_bits_left(&gb) >= 1 && !get_bits1(&gb)) {
                    mb = decode_macroblock(s, &gb, &cb_index, superblock_index);
                    insert_mb_into_sb(&sb, mb, get_bits(&gb, 4));
            copy_superblock(new_frame_data, new_stride, sb.pixels, 8);
        superblock_col_index++;
        new_frame_data += 8;
        if (old_frame_data)
            old_frame_data += 8;
        if (superblock_col_index == superblocks_per_row) {
            new_frame_data += new_stride * 8 - superblocks_per_row * 8;
            if (old_frame_data)
                old_frame_data += old_stride * 8 - superblocks_per_row * 8;
            superblock_col_index = 0;
        skip--;
    av_log(avctx, AV_LOG_DEBUG,
           "Escape sizes: %i, %i, %i\n",
           frame_size, buf_size, get_bits_count(&gb) / 8);
    av_frame_unref(s->frame);
    if ((ret = av_frame_ref(s->frame, frame)) < 0)
        return ret;
    *got_frame = 1;
    return frame_size;
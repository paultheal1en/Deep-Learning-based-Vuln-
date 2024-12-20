static int ivi_decode_blocks(GetBitContext *gb, IVIBandDesc *band, IVITile *tile,
                             AVCodecContext *avctx)
{
    int         mbn, blk, num_blocks, num_coeffs, blk_size, scan_pos, run, val,
                pos, is_intra, mc_type = 0, mv_x, mv_y, col_mask;
    uint8_t     col_flags[8];
    int32_t     prev_dc, trvec[64];
    uint32_t    cbp, sym, lo, hi, quant, buf_offs, q;
    IVIMbInfo   *mb;
    RVMapDesc   *rvmap = band->rv_map;
    void (*mc_with_delta_func)(int16_t *buf, const int16_t *ref_buf, uint32_t pitch, int mc_type);
    void (*mc_no_delta_func)  (int16_t *buf, const int16_t *ref_buf, uint32_t pitch, int mc_type);
    const uint16_t  *base_tab;
    const uint8_t   *scale_tab;
    prev_dc = 0; /* init intra prediction for the DC coefficient */
    blk_size   = band->blk_size;
    col_mask   = blk_size - 1; /* column mask for tracking non-zero coeffs */
    num_blocks = (band->mb_size != blk_size) ? 4 : 1; /* number of blocks per mb */
    num_coeffs = blk_size * blk_size;
    if (blk_size == 8) {
        mc_with_delta_func = ff_ivi_mc_8x8_delta;
        mc_no_delta_func   = ff_ivi_mc_8x8_no_delta;
    } else {
        mc_with_delta_func = ff_ivi_mc_4x4_delta;
        mc_no_delta_func   = ff_ivi_mc_4x4_no_delta;
    for (mbn = 0, mb = tile->mbs; mbn < tile->num_MBs; mb++, mbn++) {
        is_intra = !mb->type;
        cbp      = mb->cbp;
        buf_offs = mb->buf_offs;
        quant = av_clip(band->glob_quant + mb->q_delta, 0, 23);
        base_tab  = is_intra ? band->intra_base  : band->inter_base;
        scale_tab = is_intra ? band->intra_scale : band->inter_scale;
        if (scale_tab)
            quant = scale_tab[quant];
        if (!is_intra) {
            mv_x = mb->mv_x;
            mv_y = mb->mv_y;
            if (band->is_halfpel) {
                mc_type = ((mv_y & 1) << 1) | (mv_x & 1);
                mv_x >>= 1;
                mv_y >>= 1; /* convert halfpel vectors into fullpel ones */
            if (mb->type) {
                int dmv_x, dmv_y, cx, cy;
                dmv_x = mb->mv_x >> band->is_halfpel;
                dmv_y = mb->mv_y >> band->is_halfpel;
                cx    = mb->mv_x &  band->is_halfpel;
                cy    = mb->mv_y &  band->is_halfpel;
                if (   mb->xpos + dmv_x < 0
                    || mb->xpos + dmv_x + band->mb_size + cx > band->pitch
                    || mb->ypos + dmv_y < 0
                    || mb->ypos + dmv_y + band->mb_size + cy > band->aheight) {
        for (blk = 0; blk < num_blocks; blk++) {
            /* adjust block position in the buffer according to its number */
            if (blk & 1) {
                buf_offs += blk_size;
            } else if (blk == 2) {
                buf_offs -= blk_size;
                buf_offs += blk_size * band->pitch;
            if (cbp & 1) { /* block coded ? */
                scan_pos = -1;
                memset(trvec, 0, num_coeffs*sizeof(trvec[0])); /* zero transform vector */
                memset(col_flags, 0, sizeof(col_flags));      /* zero column flags */
                while (scan_pos <= num_coeffs) {
                    sym = get_vlc2(gb, band->blk_vlc.tab->table, IVI_VLC_BITS, 1);
                    if (sym == rvmap->eob_sym)
                        break; /* End of block */
                    if (sym == rvmap->esc_sym) { /* Escape - run/val explicitly coded using 3 vlc codes */
                        run = get_vlc2(gb, band->blk_vlc.tab->table, IVI_VLC_BITS, 1) + 1;
                        lo  = get_vlc2(gb, band->blk_vlc.tab->table, IVI_VLC_BITS, 1);
                        hi  = get_vlc2(gb, band->blk_vlc.tab->table, IVI_VLC_BITS, 1);
                        val = IVI_TOSIGNED((hi << 6) | lo); /* merge them and convert into signed val */
                    } else {
                        if (sym >= 256U) {
                            av_log(avctx, AV_LOG_ERROR, "Invalid sym encountered: %d.\n", sym);
                            return -1;
                        run = rvmap->runtab[sym];
                        val = rvmap->valtab[sym];
                    /* de-zigzag and dequantize */
                    scan_pos += run;
                    if (scan_pos >= num_coeffs)
                        break;
                    pos = band->scan[scan_pos];
                    if (!val)
                        av_dlog(avctx, "Val = 0 encountered!\n");
                    q = (base_tab[pos] * quant) >> 9;
                    if (q > 1)
                        val = val * q + FFSIGN(val) * (((q ^ 1) - 1) >> 1);
                    trvec[pos] = val;
                    col_flags[pos & col_mask] |= !!val; /* track columns containing non-zero coeffs */
                }// while
                if (scan_pos >= num_coeffs && sym != rvmap->eob_sym)
                    return -1; /* corrupt block data */
                /* undoing DC coeff prediction for intra-blocks */
                if (is_intra && band->is_2d_trans) {
                    prev_dc      += trvec[0];
                    trvec[0]      = prev_dc;
                    col_flags[0] |= !!prev_dc;
                /* apply inverse transform */
                band->inv_transform(trvec, band->buf + buf_offs,
                                    band->pitch, col_flags);
                /* apply motion compensation */
                if (!is_intra)
                    mc_with_delta_func(band->buf + buf_offs,
                                       band->ref_buf + buf_offs + mv_y * band->pitch + mv_x,
                                       band->pitch, mc_type);
            } else {
                /* block not coded */
                /* for intra blocks apply the dc slant transform */
                /* for inter - perform the motion compensation without delta */
                if (is_intra && band->dc_transform) {
                    band->dc_transform(&prev_dc, band->buf + buf_offs,
                                       band->pitch, blk_size);
                } else
                    mc_no_delta_func(band->buf + buf_offs,
                                     band->ref_buf + buf_offs + mv_y * band->pitch + mv_x,
                                     band->pitch, mc_type);
            cbp >>= 1;
        }// for blk
    }// for mbn
    align_get_bits(gb);
    return 0;
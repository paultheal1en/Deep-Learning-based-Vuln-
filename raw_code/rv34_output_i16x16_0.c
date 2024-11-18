static void rv34_output_i16x16(RV34DecContext *r, int8_t *intra_types, int cbp)

{

    LOCAL_ALIGNED_16(DCTELEM, block16, [16]);

    MpegEncContext *s    = &r->s;

    GetBitContext  *gb   = &s->gb;

    int             q_dc = rv34_qscale_tab[ r->luma_dc_quant_i[s->qscale] ],

                    q_ac = rv34_qscale_tab[s->qscale];

    uint8_t        *dst  = s->dest[0];

    DCTELEM        *ptr  = s->block[0];

    int       avail[6*8] = {0};

    int i, j, itype, has_ac;



    memset(block16, 0, 16 * sizeof(*block16));



    // Set neighbour information.

    if(r->avail_cache[1])

        avail[0] = 1;

    if(r->avail_cache[2])

        avail[1] = avail[2] = 1;

    if(r->avail_cache[3])

        avail[3] = avail[4] = 1;

    if(r->avail_cache[4])

        avail[5] = 1;

    if(r->avail_cache[5])

        avail[8] = avail[16] = 1;

    if(r->avail_cache[9])

        avail[24] = avail[32] = 1;



    has_ac = rv34_decode_block(block16, gb, r->cur_vlcs, 3, 0, q_dc, q_dc, q_ac);

    if(has_ac)

        r->rdsp.rv34_inv_transform(block16);

    else

        r->rdsp.rv34_inv_transform_dc(block16);



    itype = ittrans16[intra_types[0]];

    itype = adjust_pred16(itype, r->avail_cache[6-4], r->avail_cache[6-1]);

    r->h.pred16x16[itype](dst, s->linesize);



    for(j = 0; j < 4; j++){

        for(i = 0; i < 4; i++, cbp >>= 1){

            int dc = block16[i + j*4];



            if(cbp & 1){

                has_ac = rv34_decode_block(ptr, gb, r->cur_vlcs, r->luma_vlc, 0, q_ac, q_ac, q_ac);

            }else

                has_ac = 0;



            if(has_ac){

                ptr[0] = dc;

                r->rdsp.rv34_idct_add(dst+4*i, s->linesize, ptr);

            }else

                r->rdsp.rv34_idct_dc_add(dst+4*i, s->linesize, dc);

        }



        dst += 4*s->linesize;

    }



    itype = ittrans16[intra_types[0]];

    if(itype == PLANE_PRED8x8) itype = DC_PRED8x8;

    itype = adjust_pred16(itype, r->avail_cache[6-4], r->avail_cache[6-1]);



    q_dc = rv34_qscale_tab[rv34_chroma_quant[1][s->qscale]];

    q_ac = rv34_qscale_tab[rv34_chroma_quant[0][s->qscale]];



    for(j = 1; j < 3; j++){

        dst = s->dest[j];

        r->h.pred8x8[itype](dst, s->uvlinesize);

        for(i = 0; i < 4; i++, cbp >>= 1){

            uint8_t *pdst;

            if(!(cbp & 1)) continue;

            pdst   = dst + (i&1)*4 + (i&2)*2*s->uvlinesize;



            rv34_process_block(r, pdst, s->uvlinesize,

                               r->chroma_vlc, 1, q_dc, q_ac);

        }

    }

}

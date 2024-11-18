void ff_xvmc_pack_pblocks(MpegEncContext *s, int cbp)

{

    int i, j = 0;

    const int mb_block_count = 4 + (1 << s->chroma_format);



    cbp <<= 12-mb_block_count;

    for (i = 0; i < mb_block_count; i++) {

        if (cbp & (1 << 11))

            s->pblocks[i] = &s->block[j++];

        else

            s->pblocks[i] = NULL;

        cbp += cbp;

    }

}

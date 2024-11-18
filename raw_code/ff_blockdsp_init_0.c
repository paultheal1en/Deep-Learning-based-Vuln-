av_cold void ff_blockdsp_init(BlockDSPContext *c, AVCodecContext *avctx)

{

    c->clear_block  = clear_block_8_c;

    c->clear_blocks = clear_blocks_8_c;



    c->fill_block_tab[0] = fill_block16_c;

    c->fill_block_tab[1] = fill_block8_c;



    if (ARCH_ARM)

        ff_blockdsp_init_arm(c);

    if (ARCH_PPC)

        ff_blockdsp_init_ppc(c);

    if (ARCH_X86)

#if FF_API_XVMC

        ff_blockdsp_init_x86(c, avctx);

#else

        ff_blockdsp_init_x86(c);

#endif /* FF_API_XVMC */

}

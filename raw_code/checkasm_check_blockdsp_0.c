void checkasm_check_blockdsp(void)

{

    LOCAL_ALIGNED_16(uint16_t, buf0, [6 * 8 * 8]);

    LOCAL_ALIGNED_16(uint16_t, buf1, [6 * 8 * 8]);



    AVCodecContext avctx = { 0 };

    BlockDSPContext h;



    ff_blockdsp_init(&h, &avctx);



    check_clear(clear_block,  8 * 8);

    check_clear(clear_blocks, 8 * 8 * 6);



    report("blockdsp");

}

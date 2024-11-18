void ff_xvmc_init_block(MpegEncContext *s)

{

    struct xvmc_pix_fmt *render = (struct xvmc_pix_fmt*)s->current_picture.f->data[2];

    assert(render && render->xvmc_id == AV_XVMC_ID);



    s->block = (int16_t (*)[64])(render->data_blocks + render->next_free_data_block_num * 64);

}

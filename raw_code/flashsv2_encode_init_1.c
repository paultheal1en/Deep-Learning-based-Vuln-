static av_cold int flashsv2_encode_init(AVCodecContext * avctx)

{

    FlashSV2Context *s = avctx->priv_data;



    s->avctx = avctx;



    s->comp = avctx->compression_level;

    if (s->comp == -1)

        s->comp = 9;

    if (s->comp < 0 || s->comp > 9) {


               "Compression level should be 0-9, not %d\n", s->comp);







    if ((avctx->width > 4095) || (avctx->height > 4095)) {


               "Input dimensions too large, input must be max 4096x4096 !\n");










    if (av_image_check_size(avctx->width, avctx->height, 0, avctx) < 0)






    s->last_key_frame = 0;



    s->image_width  = avctx->width;

    s->image_height = avctx->height;



    s->block_width  = (s->image_width /  12) & ~15;

    s->block_height = (s->image_height / 12) & ~15;



    if(!s->block_width)

        s->block_width = 1;

    if(!s->block_height)

        s->block_height = 1;



    s->rows = (s->image_height + s->block_height - 1) / s->block_height;

    s->cols = (s->image_width +  s->block_width -  1) / s->block_width;



    s->frame_size  = s->image_width * s->image_height * 3;

    s->blocks_size = s->rows * s->cols * sizeof(Block);



    s->encbuffer     = av_mallocz(s->frame_size);

    s->keybuffer     = av_mallocz(s->frame_size);

    s->databuffer    = av_mallocz(s->frame_size * 6);

    s->current_frame = av_mallocz(s->frame_size);

    s->key_frame     = av_mallocz(s->frame_size);

    s->frame_blocks  = av_mallocz(s->blocks_size);

    s->key_blocks    = av_mallocz(s->blocks_size);



    init_blocks(s, s->frame_blocks, s->encbuffer, s->databuffer);

    init_blocks(s, s->key_blocks,   s->keybuffer, 0);

    reset_stats(s);

#ifndef FLASHSV2_DUMB

    s->total_bits = 1;

#endif



    s->use_custom_palette =  0;

    s->palette_type       = -1;        // so that the palette will be generated in reconfigure_at_keyframe



    if (!s->encbuffer || !s->keybuffer || !s->databuffer

        || !s->current_frame || !s->key_frame || !s->key_blocks

        || !s->frame_blocks) {

        av_log(avctx, AV_LOG_ERROR, "Memory allocation failed.\n");

        cleanup(s);





    return 0;

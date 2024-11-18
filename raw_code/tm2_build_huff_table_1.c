static int tm2_build_huff_table(TM2Context *ctx, TM2Codes *code)

{

    TM2Huff huff;

    int res = 0;



    huff.val_bits = get_bits(&ctx->gb, 5);

    huff.max_bits = get_bits(&ctx->gb, 5);

    huff.min_bits = get_bits(&ctx->gb, 5);

    huff.nodes = get_bits_long(&ctx->gb, 17);

    huff.num = 0;



    /* check for correct codes parameters */

    if((huff.val_bits < 1) || (huff.val_bits > 32) ||

       (huff.max_bits < 0) || (huff.max_bits > 32)) {

        av_log(ctx->avctx, AV_LOG_ERROR, "Incorrect tree parameters - literal length: %i, max code length: %i\n",

               huff.val_bits, huff.max_bits);

        return -1;

    }

    if((huff.nodes <= 0) || (huff.nodes > 0x10000)) {

        av_log(ctx->avctx, AV_LOG_ERROR, "Incorrect number of Huffman tree nodes: %i\n", huff.nodes);

        return -1;

    }

    /* one-node tree */

    if(huff.max_bits == 0)

        huff.max_bits = 1;



    /* allocate space for codes - it is exactly ceil(nodes / 2) entries */

    huff.max_num = (huff.nodes + 1) >> 1;

    huff.nums = av_mallocz(huff.max_num * sizeof(int));

    huff.bits = av_mallocz(huff.max_num * sizeof(uint32_t));

    huff.lens = av_mallocz(huff.max_num * sizeof(int));



    if(tm2_read_tree(ctx, 0, 0, &huff) == -1)

        res = -1;



    if(huff.num != huff.max_num) {

        av_log(ctx->avctx, AV_LOG_ERROR, "Got less codes than expected: %i of %i\n",

               huff.num, huff.max_num);

        res = -1;

    }



    /* convert codes to vlc_table */

    if(res != -1) {

        int i;



        res = init_vlc(&code->vlc, huff.max_bits, huff.max_num,

                    huff.lens, sizeof(int), sizeof(int),

                    huff.bits, sizeof(uint32_t), sizeof(uint32_t), 0);

        if(res < 0) {

            av_log(ctx->avctx, AV_LOG_ERROR, "Cannot build VLC table\n");

            res = -1;

        } else

            res = 0;

        if(res != -1) {

            code->bits = huff.max_bits;

            code->length = huff.max_num;

            code->recode = av_malloc(code->length * sizeof(int));

            for(i = 0; i < code->length; i++)

                code->recode[i] = huff.nums[i];

        }

    }

    /* free allocated memory */

    av_free(huff.nums);

    av_free(huff.bits);

    av_free(huff.lens);



    return res;

}

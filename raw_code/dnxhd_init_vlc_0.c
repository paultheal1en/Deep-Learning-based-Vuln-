static int dnxhd_init_vlc(DNXHDEncContext *ctx)

{

    int i, j, level, run;

    int max_level = 1<<(ctx->cid_table->bit_depth+2);



    FF_ALLOCZ_OR_GOTO(ctx->m.avctx, ctx->vlc_codes, max_level*4*sizeof(*ctx->vlc_codes), fail);

    FF_ALLOCZ_OR_GOTO(ctx->m.avctx, ctx->vlc_bits,  max_level*4*sizeof(*ctx->vlc_bits) , fail);

    FF_ALLOCZ_OR_GOTO(ctx->m.avctx, ctx->run_codes, 63*2,                                fail);

    FF_ALLOCZ_OR_GOTO(ctx->m.avctx, ctx->run_bits,  63,                                  fail);



    ctx->vlc_codes += max_level*2;

    ctx->vlc_bits  += max_level*2;

    for (level = -max_level; level < max_level; level++) {

        for (run = 0; run < 2; run++) {

            int index = (level<<1)|run;

            int sign, offset = 0, alevel = level;



            MASK_ABS(sign, alevel);

            if (alevel > 64) {

                offset = (alevel-1)>>6;

                alevel -= offset<<6;

            }

            for (j = 0; j < 257; j++) {

                if (ctx->cid_table->ac_level[j] >> 1 == alevel &&

                    (!offset || (ctx->cid_table->ac_index_flag[j] && offset)) &&

                    (!run    || (ctx->cid_table->ac_run_flag  [j] && run))) {

                    assert(!ctx->vlc_codes[index]);

                    if (alevel) {

                        ctx->vlc_codes[index] = (ctx->cid_table->ac_codes[j]<<1)|(sign&1);

                        ctx->vlc_bits [index] = ctx->cid_table->ac_bits[j]+1;

                    } else {

                        ctx->vlc_codes[index] = ctx->cid_table->ac_codes[j];

                        ctx->vlc_bits [index] = ctx->cid_table->ac_bits [j];

                    }

                    break;

                }

            }

            assert(!alevel || j < 257);

            if (offset) {

                ctx->vlc_codes[index] = (ctx->vlc_codes[index]<<ctx->cid_table->index_bits)|offset;

                ctx->vlc_bits [index]+= ctx->cid_table->index_bits;

            }

        }

    }

    for (i = 0; i < 62; i++) {

        int run = ctx->cid_table->run[i];

        assert(run < 63);

        ctx->run_codes[run] = ctx->cid_table->run_codes[i];

        ctx->run_bits [run] = ctx->cid_table->run_bits[i];

    }

    return 0;

 fail:

    return -1;

}

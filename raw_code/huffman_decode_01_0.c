static int huffman_decode(MPADecodeContext *s, GranuleDef *g,

                          int16_t *exponents, int end_pos)

{

    int s_index;

    int linbits, code, x, y, l, v, i, j, k, pos;

    int last_pos;

    VLC *vlc;



    /* low frequencies (called big values) */

    s_index = 0;

    for(i=0;i<3;i++) {

        j = g->region_size[i];

        if (j == 0)

            continue;

        /* select vlc table */

        k = g->table_select[i];

        l = mpa_huff_data[k][0];

        linbits = mpa_huff_data[k][1];

        vlc = &huff_vlc[l];



        if(!l){

            memset(&g->sb_hybrid[s_index], 0, sizeof(*g->sb_hybrid)*j);

            s_index += 2*j;

            continue;

        }



        /* read huffcode and compute each couple */

        for(;j>0;j--) {

            int exponent;



            if (get_bits_count(&s->gb) >= end_pos)

                break;

            y = get_vlc2(&s->gb, vlc->table, 7, 3);



            if(!y){

                g->sb_hybrid[s_index  ] =

                g->sb_hybrid[s_index+1] = 0;

                s_index += 2;

                continue;

            }



            x = y >> 4;

            y = y & 0x0f;

            exponent= exponents[s_index];



            dprintf("region=%d n=%d x=%d y=%d exp=%d\n",

                    i, g->region_size[i] - j, x, y, exponent);

            if (x) {

#if 0

                if (x == 15)

                    x += get_bitsz(&s->gb, linbits);

                v = l3_unscale(x, exponent);

#else

                if (x < 15){

                    v = expval_table[ exponent + 400 ][ x ];

//                      v = expval_table[ (exponent&3) + 400 ][ x ] >> FFMIN(0 - (exponent>>2), 31);

                }else{

                    x += get_bitsz(&s->gb, linbits);

                    v = l3_unscale(x, exponent);

                }

#endif

                if (get_bits1(&s->gb))

                    v = -v;

            } else {

                v = 0;

            }

            g->sb_hybrid[s_index++] = v;

            if (y) {

#if 0

                if (y == 15)

                    y += get_bitsz(&s->gb, linbits);

                v = l3_unscale(y, exponent);

#else

                if (y < 15){

                    v = expval_table[ exponent + 400 ][ y ];

                }else{

                    y += get_bitsz(&s->gb, linbits);

                    v = l3_unscale(y, exponent);

                }

#endif

                if (get_bits1(&s->gb))

                    v = -v;

            } else {

                v = 0;

            }

            g->sb_hybrid[s_index++] = v;

        }

    }



    /* high frequencies */

    vlc = &huff_quad_vlc[g->count1table_select];

    last_pos=0;

    while (s_index <= 572) {

        pos = get_bits_count(&s->gb);

        if (pos >= end_pos) {

            if (pos > end_pos && last_pos){

                /* some encoders generate an incorrect size for this

                   part. We must go back into the data */

                s_index -= 4;

                init_get_bits(&s->gb, s->gb.buffer + 4*(last_pos>>5), s->gb.size_in_bits - (last_pos&(~31)));

                skip_bits(&s->gb, last_pos&31);

            }

            break;

        }

        last_pos= pos;



        code = get_vlc2(&s->gb, vlc->table, vlc->bits, 1);

        dprintf("t=%d code=%d\n", g->count1table_select, code);

        g->sb_hybrid[s_index+0]=

        g->sb_hybrid[s_index+1]=

        g->sb_hybrid[s_index+2]=

        g->sb_hybrid[s_index+3]= 0;

        while(code){

            const static int idxtab[16]={3,3,2,2,1,1,1,1,0,0,0,0,0,0,0,0};

            int pos= s_index+idxtab[code];

            code ^= 8>>idxtab[code];

            v = exp_table[ exponents[pos] + 400];

            if(get_bits1(&s->gb))

                v = -v;

            g->sb_hybrid[pos] = v;

        }

        s_index+=4;

    }

    memset(&g->sb_hybrid[s_index], 0, sizeof(*g->sb_hybrid)*(576 - s_index));

    return 0;

}

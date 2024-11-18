static void vp6_parse_coeff_models(VP56Context *s)

{

    VP56RangeCoder *c = &s->c;

    VP56Model *model = s->modelp;

    int def_prob[11];

    int node, cg, ctx, pos;

    int ct;    /* code type */

    int pt;    /* plane type (0 for Y, 1 for U or V) */



    memset(def_prob, 0x80, sizeof(def_prob));



    for (pt=0; pt<2; pt++)

        for (node=0; node<11; node++)

            if (vp56_rac_get_prob(c, vp6_dccv_pct[pt][node])) {

                def_prob[node] = vp56_rac_gets_nn(c, 7);

                model->coeff_dccv[pt][node] = def_prob[node];

            } else if (s->framep[VP56_FRAME_CURRENT]->key_frame) {

                model->coeff_dccv[pt][node] = def_prob[node];

            }



    if (vp56_rac_get(c)) {

        for (pos=1; pos<64; pos++)

            if (vp56_rac_get_prob(c, vp6_coeff_reorder_pct[pos]))

                model->coeff_reorder[pos] = vp56_rac_gets(c, 4);

        vp6_coeff_order_table_init(s);

    }



    for (cg=0; cg<2; cg++)

        for (node=0; node<14; node++)

            if (vp56_rac_get_prob(c, vp6_runv_pct[cg][node]))

                model->coeff_runv[cg][node] = vp56_rac_gets_nn(c, 7);



    for (ct=0; ct<3; ct++)

        for (pt=0; pt<2; pt++)

            for (cg=0; cg<6; cg++)

                for (node=0; node<11; node++)

                    if (vp56_rac_get_prob(c, vp6_ract_pct[ct][pt][cg][node])) {

                        def_prob[node] = vp56_rac_gets_nn(c, 7);

                        model->coeff_ract[pt][ct][cg][node] = def_prob[node];

                    } else if (s->framep[VP56_FRAME_CURRENT]->key_frame) {

                        model->coeff_ract[pt][ct][cg][node] = def_prob[node];

                    }



    if (s->use_huffman) {

        for (pt=0; pt<2; pt++) {

            vp6_build_huff_tree(s, model->coeff_dccv[pt],

                                vp6_huff_coeff_map, 12, &s->dccv_vlc[pt]);

            vp6_build_huff_tree(s, model->coeff_runv[pt],

                                vp6_huff_run_map, 9, &s->runv_vlc[pt]);

            for (ct=0; ct<3; ct++)

                for (cg = 0; cg < 6; cg++)

                    vp6_build_huff_tree(s, model->coeff_ract[pt][ct][cg],

                                        vp6_huff_coeff_map, 12,

                                        &s->ract_vlc[pt][ct][cg]);

        }

        memset(s->nb_null, 0, sizeof(s->nb_null));

    } else {

    /* coeff_dcct is a linear combination of coeff_dccv */

    for (pt=0; pt<2; pt++)

        for (ctx=0; ctx<3; ctx++)

            for (node=0; node<5; node++)

                model->coeff_dcct[pt][ctx][node] = av_clip(((model->coeff_dccv[pt][node] * vp6_dccv_lc[ctx][node][0] + 128) >> 8) + vp6_dccv_lc[ctx][node][1], 1, 255);

    }

}

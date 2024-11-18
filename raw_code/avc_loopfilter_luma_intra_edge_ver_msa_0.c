static void avc_loopfilter_luma_intra_edge_ver_msa(uint8_t *data,

                                                   uint8_t alpha_in,

                                                   uint8_t beta_in,

                                                   uint32_t img_width)

{

    uint8_t *src;

    v16u8 alpha, beta, p0_asub_q0;

    v16u8 is_less_than_alpha, is_less_than;

    v16u8 is_less_than_beta, negate_is_less_than_beta;

    v8i16 p2_r = { 0 };

    v8i16 p1_r = { 0 };

    v8i16 p0_r = { 0 };

    v8i16 q0_r = { 0 };

    v8i16 q1_r = { 0 };

    v8i16 q2_r = { 0 };

    v8i16 p2_l = { 0 };

    v8i16 p1_l = { 0 };

    v8i16 p0_l = { 0 };

    v8i16 q0_l = { 0 };

    v8i16 q1_l = { 0 };

    v8i16 q2_l = { 0 };

    v16u8 p3_org, p2_org, p1_org, p0_org, q0_org, q1_org, q2_org, q3_org;

    v8i16 p1_org_r, p0_org_r, q0_org_r, q1_org_r;

    v8i16 p1_org_l, p0_org_l, q0_org_l, q1_org_l;

    v16i8 zero = { 0 };

    v16u8 tmp_flag;



    src = data - 4;



    {

        v16u8 row0, row1, row2, row3, row4, row5, row6, row7;

        v16u8 row8, row9, row10, row11, row12, row13, row14, row15;



        LOAD_8VECS_UB(src, img_width,

                      row0, row1, row2, row3, row4, row5, row6, row7);

        LOAD_8VECS_UB(src + (8 * img_width), img_width,

                      row8, row9, row10, row11, row12, row13, row14, row15);



        TRANSPOSE16x8_B_UB(row0, row1, row2, row3, row4, row5, row6, row7,

                           row8, row9, row10, row11, row12, row13, row14, row15,

                           p3_org, p2_org, p1_org, p0_org,

                           q0_org, q1_org, q2_org, q3_org);

    }



    p1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p1_org);

    p0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p0_org);

    q0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q0_org);

    q1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q1_org);



    p1_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p1_org);

    p0_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p0_org);

    q0_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q0_org);

    q1_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q1_org);



    /*  if ( ((unsigned)ABS(p0-q0) < thresholds->alpha_in) &&

       ((unsigned)ABS(p1-p0) < thresholds->beta_in)  &&

       ((unsigned)ABS(q1-q0) < thresholds->beta_in) )   */

    {

        v16u8 p1_asub_p0, q1_asub_q0;



        p0_asub_q0 = __msa_asub_u_b(p0_org, q0_org);

        p1_asub_p0 = __msa_asub_u_b(p1_org, p0_org);

        q1_asub_q0 = __msa_asub_u_b(q1_org, q0_org);



        alpha = (v16u8) __msa_fill_b(alpha_in);

        beta = (v16u8) __msa_fill_b(beta_in);



        is_less_than_alpha = (p0_asub_q0 < alpha);

        is_less_than_beta = (p1_asub_p0 < beta);

        is_less_than = is_less_than_beta & is_less_than_alpha;

        is_less_than_beta = (q1_asub_q0 < beta);

        is_less_than = is_less_than_beta & is_less_than;

    }



    if (!__msa_test_bz_v(is_less_than)) {

        tmp_flag = alpha >> 2;

        tmp_flag = tmp_flag + 2;

        tmp_flag = (p0_asub_q0 < tmp_flag);



        {

            v16u8 p2_asub_p0;



            p2_asub_p0 = __msa_asub_u_b(p2_org, p0_org);

            is_less_than_beta = (p2_asub_p0 < beta);

        }



        is_less_than_beta = tmp_flag & is_less_than_beta;



        negate_is_less_than_beta = __msa_xori_b(is_less_than_beta, 0xff);

        is_less_than_beta = is_less_than_beta & is_less_than;

        negate_is_less_than_beta = negate_is_less_than_beta & is_less_than;



        /* right */

        {

            v16u8 is_less_than_beta_r;



            is_less_than_beta_r =

                (v16u8) __msa_sldi_b((v16i8) is_less_than_beta, zero, 8);

            if (!__msa_test_bz_v(is_less_than_beta_r)) {

                v8i16 p3_org_r;



                p3_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p3_org);

                p2_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p2_org);



                AVC_LOOP_FILTER_P0P1P2_OR_Q0Q1Q2(p3_org_r, p0_org_r,

                                                 q0_org_r, p1_org_r,

                                                 p2_r, q1_org_r,

                                                 p0_r, p1_r, p2_r);

            }

        }

        /* left */

        {

            v16u8 is_less_than_beta_l;



            is_less_than_beta_l =

                (v16u8) __msa_sldi_b(zero, (v16i8) is_less_than_beta, 8);

            if (!__msa_test_bz_v(is_less_than_beta_l)) {

                v8i16 p3_org_l;



                p3_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p3_org);

                p2_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p2_org);



                AVC_LOOP_FILTER_P0P1P2_OR_Q0Q1Q2(p3_org_l, p0_org_l,

                                                 q0_org_l, p1_org_l,

                                                 p2_l, q1_org_l,

                                                 p0_l, p1_l, p2_l);

            }

        }



        /* combine and store */

        if (!__msa_test_bz_v(is_less_than_beta)) {

            v16u8 p0, p2, p1;



            p0 = (v16u8) __msa_pckev_b((v16i8) p0_l, (v16i8) p0_r);

            p1 = (v16u8) __msa_pckev_b((v16i8) p1_l, (v16i8) p1_r);

            p2 = (v16u8) __msa_pckev_b((v16i8) p2_l, (v16i8) p2_r);



            p0_org = __msa_bmnz_v(p0_org, p0, is_less_than_beta);

            p1_org = __msa_bmnz_v(p1_org, p1, is_less_than_beta);

            p2_org = __msa_bmnz_v(p2_org, p2, is_less_than_beta);

        }



        /* right */

        {

            v16u8 negate_is_less_than_beta_r;



            negate_is_less_than_beta_r =

                (v16u8) __msa_sldi_b((v16i8) negate_is_less_than_beta, zero, 8);



            if (!__msa_test_bz_v(negate_is_less_than_beta_r)) {

                AVC_LOOP_FILTER_P0_OR_Q0(p0_org_r, q1_org_r, p1_org_r, p0_r);

            }

        }



        /* left */

        {

            v16u8 negate_is_less_than_beta_l;



            negate_is_less_than_beta_l =

                (v16u8) __msa_sldi_b(zero, (v16i8) negate_is_less_than_beta, 8);

            if (!__msa_test_bz_v(negate_is_less_than_beta_l)) {

                AVC_LOOP_FILTER_P0_OR_Q0(p0_org_l, q1_org_l, p1_org_l, p0_l);

            }

        }



        if (!__msa_test_bz_v(negate_is_less_than_beta)) {

            v16u8 p0;



            p0 = (v16u8) __msa_pckev_b((v16i8) p0_l, (v16i8) p0_r);

            p0_org = __msa_bmnz_v(p0_org, p0, negate_is_less_than_beta);

        }



        {

            v16u8 q2_asub_q0;



            q2_asub_q0 = __msa_asub_u_b(q2_org, q0_org);

            is_less_than_beta = (q2_asub_q0 < beta);

        }



        is_less_than_beta = is_less_than_beta & tmp_flag;

        negate_is_less_than_beta = __msa_xori_b(is_less_than_beta, 0xff);



        is_less_than_beta = is_less_than_beta & is_less_than;

        negate_is_less_than_beta = negate_is_less_than_beta & is_less_than;



        /* right */

        {

            v16u8 is_less_than_beta_r;



            is_less_than_beta_r =

                (v16u8) __msa_sldi_b((v16i8) is_less_than_beta, zero, 8);

            if (!__msa_test_bz_v(is_less_than_beta_r)) {

                v8i16 q3_org_r;



                q3_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q3_org);

                q2_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q2_org);



                AVC_LOOP_FILTER_P0P1P2_OR_Q0Q1Q2(q3_org_r, q0_org_r,

                                                 p0_org_r, q1_org_r,

                                                 q2_r, p1_org_r,

                                                 q0_r, q1_r, q2_r);

            }

        }



        /* left */

        {

            v16u8 is_less_than_beta_l;



            is_less_than_beta_l =

                (v16u8) __msa_sldi_b(zero, (v16i8) is_less_than_beta, 8);

            if (!__msa_test_bz_v(is_less_than_beta_l)) {

                v8i16 q3_org_l;



                q3_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q3_org);

                q2_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q2_org);



                AVC_LOOP_FILTER_P0P1P2_OR_Q0Q1Q2(q3_org_l, q0_org_l,

                                                 p0_org_l, q1_org_l,

                                                 q2_l, p1_org_l,

                                                 q0_l, q1_l, q2_l);

            }

        }



        /* combine and store */

        if (!__msa_test_bz_v(is_less_than_beta)) {

            v16u8 q0, q1, q2;



            q0 = (v16u8) __msa_pckev_b((v16i8) q0_l, (v16i8) q0_r);

            q1 = (v16u8) __msa_pckev_b((v16i8) q1_l, (v16i8) q1_r);

            q2 = (v16u8) __msa_pckev_b((v16i8) q2_l, (v16i8) q2_r);



            q0_org = __msa_bmnz_v(q0_org, q0, is_less_than_beta);

            q1_org = __msa_bmnz_v(q1_org, q1, is_less_than_beta);

            q2_org = __msa_bmnz_v(q2_org, q2, is_less_than_beta);

        }



        /* right */

        {

            v16u8 negate_is_less_than_beta_r;



            negate_is_less_than_beta_r =

                (v16u8) __msa_sldi_b((v16i8) negate_is_less_than_beta, zero, 8);

            if (!__msa_test_bz_v(negate_is_less_than_beta_r)) {

                AVC_LOOP_FILTER_P0_OR_Q0(q0_org_r, p1_org_r, q1_org_r, q0_r);

            }

        }

        /* left */

        {

            v16u8 negate_is_less_than_beta_l;



            negate_is_less_than_beta_l =

                (v16u8) __msa_sldi_b(zero, (v16i8) negate_is_less_than_beta, 8);

            if (!__msa_test_bz_v(negate_is_less_than_beta_l)) {

                AVC_LOOP_FILTER_P0_OR_Q0(q0_org_l, p1_org_l, q1_org_l, q0_l);

            }

        }



        if (!__msa_test_bz_v(negate_is_less_than_beta)) {

            v16u8 q0;



            q0 = (v16u8) __msa_pckev_b((v16i8) q0_l, (v16i8) q0_r);

            q0_org = __msa_bmnz_v(q0_org, q0, negate_is_less_than_beta);

        }

    }



    {

        v16u8 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

        uint32_t out0, out2;

        uint16_t out1, out3;



        tmp0 = (v16u8) __msa_ilvr_b((v16i8) p1_org, (v16i8) p2_org);

        tmp1 = (v16u8) __msa_ilvr_b((v16i8) q0_org, (v16i8) p0_org);

        tmp2 = (v16u8) __msa_ilvr_b((v16i8) q2_org, (v16i8) q1_org);

        tmp3 = (v16u8) __msa_ilvr_h((v8i16) tmp1, (v8i16) tmp0);

        tmp4 = (v16u8) __msa_ilvl_h((v8i16) tmp1, (v8i16) tmp0);



        tmp0 = (v16u8) __msa_ilvl_b((v16i8) p1_org, (v16i8) p2_org);

        tmp1 = (v16u8) __msa_ilvl_b((v16i8) q0_org, (v16i8) p0_org);

        tmp5 = (v16u8) __msa_ilvl_b((v16i8) q2_org, (v16i8) q1_org);

        tmp6 = (v16u8) __msa_ilvr_h((v8i16) tmp1, (v8i16) tmp0);

        tmp7 = (v16u8) __msa_ilvl_h((v8i16) tmp1, (v8i16) tmp0);



        src = data - 3;



        out0 = __msa_copy_u_w((v4i32) tmp3, 0);

        out1 = __msa_copy_u_h((v8i16) tmp2, 0);

        out2 = __msa_copy_u_w((v4i32) tmp3, 1);

        out3 = __msa_copy_u_h((v8i16) tmp2, 1);



        STORE_WORD(src, out0);

        STORE_HWORD((src + 4), out1);

        src += img_width;

        STORE_WORD(src, out2);

        STORE_HWORD((src + 4), out3);



        out0 = __msa_copy_u_w((v4i32) tmp3, 2);

        out1 = __msa_copy_u_h((v8i16) tmp2, 2);

        out2 = __msa_copy_u_w((v4i32) tmp3, 3);

        out3 = __msa_copy_u_h((v8i16) tmp2, 3);



        src += img_width;

        STORE_WORD(src, out0);

        STORE_HWORD((src + 4), out1);

        src += img_width;

        STORE_WORD(src, out2);

        STORE_HWORD((src + 4), out3);



        out0 = __msa_copy_u_w((v4i32) tmp4, 0);

        out1 = __msa_copy_u_h((v8i16) tmp2, 4);

        out2 = __msa_copy_u_w((v4i32) tmp4, 1);

        out3 = __msa_copy_u_h((v8i16) tmp2, 5);



        src += img_width;

        STORE_WORD(src, out0);

        STORE_HWORD((src + 4), out1);

        src += img_width;

        STORE_WORD(src, out2);

        STORE_HWORD((src + 4), out3);



        out0 = __msa_copy_u_w((v4i32) tmp4, 2);

        out1 = __msa_copy_u_h((v8i16) tmp2, 6);

        out2 = __msa_copy_u_w((v4i32) tmp4, 3);

        out3 = __msa_copy_u_h((v8i16) tmp2, 7);



        src += img_width;

        STORE_WORD(src, out0);

        STORE_HWORD((src + 4), out1);

        src += img_width;

        STORE_WORD(src, out2);

        STORE_HWORD((src + 4), out3);



        out0 = __msa_copy_u_w((v4i32) tmp6, 0);

        out1 = __msa_copy_u_h((v8i16) tmp5, 0);

        out2 = __msa_copy_u_w((v4i32) tmp6, 1);

        out3 = __msa_copy_u_h((v8i16) tmp5, 1);



        src += img_width;

        STORE_WORD(src, out0);

        STORE_HWORD((src + 4), out1);

        src += img_width;

        STORE_WORD(src, out2);

        STORE_HWORD((src + 4), out3);



        out0 = __msa_copy_u_w((v4i32) tmp6, 2);

        out1 = __msa_copy_u_h((v8i16) tmp5, 2);

        out2 = __msa_copy_u_w((v4i32) tmp6, 3);

        out3 = __msa_copy_u_h((v8i16) tmp5, 3);



        src += img_width;

        STORE_WORD(src, out0);

        STORE_HWORD((src + 4), out1);

        src += img_width;

        STORE_WORD(src, out2);

        STORE_HWORD((src + 4), out3);



        out0 = __msa_copy_u_w((v4i32) tmp7, 0);

        out1 = __msa_copy_u_h((v8i16) tmp5, 4);

        out2 = __msa_copy_u_w((v4i32) tmp7, 1);

        out3 = __msa_copy_u_h((v8i16) tmp5, 5);



        src += img_width;

        STORE_WORD(src, out0);

        STORE_HWORD((src + 4), out1);

        src += img_width;

        STORE_WORD(src, out2);

        STORE_HWORD((src + 4), out3);



        out0 = __msa_copy_u_w((v4i32) tmp7, 2);

        out1 = __msa_copy_u_h((v8i16) tmp5, 6);

        out2 = __msa_copy_u_w((v4i32) tmp7, 3);

        out3 = __msa_copy_u_h((v8i16) tmp5, 7);



        src += img_width;

        STORE_WORD(src, out0);

        STORE_HWORD((src + 4), out1);

        src += img_width;

        STORE_WORD(src, out2);

        STORE_HWORD((src + 4), out3);

    }

}

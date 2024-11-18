static void avc_loopfilter_cb_or_cr_inter_edge_ver_msa(uint8_t *data,

                                                       uint8_t bs0, uint8_t bs1,

                                                       uint8_t bs2, uint8_t bs3,

                                                       uint8_t tc0, uint8_t tc1,

                                                       uint8_t tc2, uint8_t tc3,

                                                       uint8_t alpha_in,

                                                       uint8_t beta_in,

                                                       uint32_t img_width)

{

    uint8_t *src;

    uint16_t out0, out1, out2, out3;

    v16u8 alpha, beta;

    v16u8 p0_asub_q0, p1_asub_p0, q1_asub_q0;

    v16u8 is_less_than, is_less_than_beta, is_less_than_alpha;

    v16u8 p0, q0;

    v8i16 p0_r = { 0 };

    v8i16 q0_r = { 0 };

    v16u8 p1_org, p0_org, q0_org, q1_org;

    v8i16 p1_org_r, p0_org_r, q0_org_r, q1_org_r;

    v16u8 is_bs_greater_than0;

    v8i16 tc_r, negate_tc_r;

    v16i8 negate_tc, sign_negate_tc;

    v16i8 zero = { 0 };

    v16u8 row0, row1, row2, row3, row4, row5, row6, row7;

    v8i16 tmp1, tmp_vec, bs = { 0 };

    v8i16 tc = { 0 };



    tmp_vec = (v8i16) __msa_fill_b(bs0);

    bs = __msa_insve_h(bs, 0, tmp_vec);

    tmp_vec = (v8i16) __msa_fill_b(bs1);

    bs = __msa_insve_h(bs, 1, tmp_vec);

    tmp_vec = (v8i16) __msa_fill_b(bs2);

    bs = __msa_insve_h(bs, 2, tmp_vec);

    tmp_vec = (v8i16) __msa_fill_b(bs3);

    bs = __msa_insve_h(bs, 3, tmp_vec);



    if (!__msa_test_bz_v((v16u8) bs)) {

        tmp_vec = (v8i16) __msa_fill_b(tc0);

        tc = __msa_insve_h(tc, 0, tmp_vec);

        tmp_vec = (v8i16) __msa_fill_b(tc1);

        tc = __msa_insve_h(tc, 1, tmp_vec);

        tmp_vec = (v8i16) __msa_fill_b(tc2);

        tc = __msa_insve_h(tc, 2, tmp_vec);

        tmp_vec = (v8i16) __msa_fill_b(tc3);

        tc = __msa_insve_h(tc, 3, tmp_vec);



        is_bs_greater_than0 = (v16u8) (zero < (v16i8) bs);



        LOAD_8VECS_UB((data - 2), img_width,

                      row0, row1, row2, row3, row4, row5, row6, row7);



        TRANSPOSE8x4_B_UB(row0, row1, row2, row3,

                          row4, row5, row6, row7,

                          p1_org, p0_org, q0_org, q1_org);



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

        is_less_than = is_bs_greater_than0 & is_less_than;



        is_less_than = (v16u8) __msa_ilvr_d((v2i64) zero, (v2i64) is_less_than);



        if (!__msa_test_bz_v(is_less_than)) {

            p1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p1_org);

            p0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p0_org);

            q0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q0_org);

            q1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q1_org);



            negate_tc = zero - (v16i8) tc;

            sign_negate_tc = __msa_clti_s_b(negate_tc, 0);



            negate_tc_r = (v8i16) __msa_ilvr_b(sign_negate_tc, negate_tc);



            tc_r = (v8i16) __msa_ilvr_b(zero, (v16i8) tc);



            AVC_LOOP_FILTER_P0Q0(q0_org_r, p0_org_r, p1_org_r, q1_org_r,

                                 negate_tc_r, tc_r, p0_r, q0_r);



            p0 = (v16u8) __msa_pckev_b(zero, (v16i8) p0_r);

            q0 = (v16u8) __msa_pckev_b(zero, (v16i8) q0_r);



            p0_org = __msa_bmnz_v(p0_org, p0, is_less_than);

            q0_org = __msa_bmnz_v(q0_org, q0, is_less_than);



            tmp1 = (v8i16) __msa_ilvr_b((v16i8) q0_org, (v16i8) p0_org);



            src = data - 1;



            out0 = __msa_copy_u_h(tmp1, 0);

            out1 = __msa_copy_u_h(tmp1, 1);

            out2 = __msa_copy_u_h(tmp1, 2);

            out3 = __msa_copy_u_h(tmp1, 3);



            STORE_HWORD(src, out0);

            src += img_width;

            STORE_HWORD(src, out1);

            src += img_width;

            STORE_HWORD(src, out2);

            src += img_width;

            STORE_HWORD(src, out3);



            out0 = __msa_copy_u_h(tmp1, 4);

            out1 = __msa_copy_u_h(tmp1, 5);

            out2 = __msa_copy_u_h(tmp1, 6);

            out3 = __msa_copy_u_h(tmp1, 7);



            src += img_width;

            STORE_HWORD(src, out0);

            src += img_width;

            STORE_HWORD(src, out1);

            src += img_width;

            STORE_HWORD(src, out2);

            src += img_width;

            STORE_HWORD(src, out3);

        }

    }

}

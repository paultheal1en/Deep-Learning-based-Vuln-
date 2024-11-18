static void avc_loopfilter_luma_inter_edge_hor_msa(uint8_t *data,

                                                   uint8_t bs0, uint8_t bs1,

                                                   uint8_t bs2, uint8_t bs3,

                                                   uint8_t tc0, uint8_t tc1,

                                                   uint8_t tc2, uint8_t tc3,

                                                   uint8_t alpha_in,

                                                   uint8_t beta_in,

                                                   uint32_t image_width)

{

    v16u8 p2_asub_p0, u8_q2asub_q0;

    v16u8 alpha, beta, is_less_than, is_less_than_beta;

    v16u8 p1, p0, q0, q1;

    v8i16 p1_r = { 0 };

    v8i16 p0_r, q0_r, q1_r = { 0 };

    v8i16 p1_l = { 0 };

    v8i16 p0_l, q0_l, q1_l = { 0 };

    v16u8 p2_org, p1_org, p0_org, q0_org, q1_org, q2_org;

    v8i16 p2_org_r, p1_org_r, p0_org_r, q0_org_r, q1_org_r, q2_org_r;

    v8i16 p2_org_l, p1_org_l, p0_org_l, q0_org_l, q1_org_l, q2_org_l;

    v16i8 zero = { 0 };

    v16u8 tmp_vec;

    v16u8 bs = { 0 };

    v16i8 tc = { 0 };



    tmp_vec = (v16u8) __msa_fill_b(bs0);

    bs = (v16u8) __msa_insve_w((v4i32) bs, 0, (v4i32) tmp_vec);

    tmp_vec = (v16u8) __msa_fill_b(bs1);

    bs = (v16u8) __msa_insve_w((v4i32) bs, 1, (v4i32) tmp_vec);

    tmp_vec = (v16u8) __msa_fill_b(bs2);

    bs = (v16u8) __msa_insve_w((v4i32) bs, 2, (v4i32) tmp_vec);

    tmp_vec = (v16u8) __msa_fill_b(bs3);

    bs = (v16u8) __msa_insve_w((v4i32) bs, 3, (v4i32) tmp_vec);



    if (!__msa_test_bz_v(bs)) {

        tmp_vec = (v16u8) __msa_fill_b(tc0);

        tc = (v16i8) __msa_insve_w((v4i32) tc, 0, (v4i32) tmp_vec);

        tmp_vec = (v16u8) __msa_fill_b(tc1);

        tc = (v16i8) __msa_insve_w((v4i32) tc, 1, (v4i32) tmp_vec);

        tmp_vec = (v16u8) __msa_fill_b(tc2);

        tc = (v16i8) __msa_insve_w((v4i32) tc, 2, (v4i32) tmp_vec);

        tmp_vec = (v16u8) __msa_fill_b(tc3);

        tc = (v16i8) __msa_insve_w((v4i32) tc, 3, (v4i32) tmp_vec);



        alpha = (v16u8) __msa_fill_b(alpha_in);

        beta = (v16u8) __msa_fill_b(beta_in);



        p2_org = LOAD_UB(data - (3 * image_width));

        p1_org = LOAD_UB(data - (image_width << 1));

        p0_org = LOAD_UB(data - image_width);

        q0_org = LOAD_UB(data);

        q1_org = LOAD_UB(data + image_width);



        {

            v16u8 p0_asub_q0, p1_asub_p0, q1_asub_q0;

            v16u8 is_less_than_alpha, is_bs_greater_than0;



            is_bs_greater_than0 = ((v16u8) zero < bs);

            p0_asub_q0 = __msa_asub_u_b(p0_org, q0_org);

            p1_asub_p0 = __msa_asub_u_b(p1_org, p0_org);

            q1_asub_q0 = __msa_asub_u_b(q1_org, q0_org);



            is_less_than_alpha = (p0_asub_q0 < alpha);

            is_less_than_beta = (p1_asub_p0 < beta);

            is_less_than = is_less_than_beta & is_less_than_alpha;

            is_less_than_beta = (q1_asub_q0 < beta);

            is_less_than = is_less_than_beta & is_less_than;

            is_less_than = is_less_than & is_bs_greater_than0;

        }



        if (!__msa_test_bz_v(is_less_than)) {

            v16i8 sign_negate_tc, negate_tc;

            v8i16 negate_tc_r, i16_negatetc_l, tc_l, tc_r;



            q2_org = LOAD_UB(data + (2 * image_width));



            negate_tc = zero - tc;

            sign_negate_tc = __msa_clti_s_b(negate_tc, 0);



            negate_tc_r = (v8i16) __msa_ilvr_b(sign_negate_tc, negate_tc);

            i16_negatetc_l = (v8i16) __msa_ilvl_b(sign_negate_tc, negate_tc);



            tc_r = (v8i16) __msa_ilvr_b(zero, tc);

            tc_l = (v8i16) __msa_ilvl_b(zero, tc);



            p1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p1_org);

            p0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p0_org);

            q0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q0_org);



            p1_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p1_org);

            p0_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p0_org);

            q0_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q0_org);



            p2_asub_p0 = __msa_asub_u_b(p2_org, p0_org);

            is_less_than_beta = (p2_asub_p0 < beta);

            is_less_than_beta = is_less_than_beta & is_less_than;



            {

                v8u16 is_less_than_beta_r, is_less_than_beta_l;



                is_less_than_beta_r =

                    (v8u16) __msa_sldi_b((v16i8) is_less_than_beta, zero, 8);



                if (!__msa_test_bz_v((v16u8) is_less_than_beta_r)) {

                    p2_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p2_org);



                    AVC_LOOP_FILTER_P1_OR_Q1(p0_org_r, q0_org_r,

                                             p1_org_r, p2_org_r,

                                             negate_tc_r, tc_r, p1_r);

                }



                is_less_than_beta_l =

                    (v8u16) __msa_sldi_b(zero, (v16i8) is_less_than_beta, 8);

                if (!__msa_test_bz_v((v16u8) is_less_than_beta_l)) {

                    p2_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p2_org);



                    AVC_LOOP_FILTER_P1_OR_Q1(p0_org_l, q0_org_l,

                                             p1_org_l, p2_org_l,

                                             i16_negatetc_l, tc_l, p1_l);

                }

            }



            if (!__msa_test_bz_v(is_less_than_beta)) {

                p1 = (v16u8) __msa_pckev_b((v16i8) p1_l, (v16i8) p1_r);

                p1_org = __msa_bmnz_v(p1_org, p1, is_less_than_beta);

                STORE_UB(p1_org, data - (2 * image_width));



                is_less_than_beta = __msa_andi_b(is_less_than_beta, 1);

                tc = tc + (v16i8) is_less_than_beta;

            }



            u8_q2asub_q0 = __msa_asub_u_b(q2_org, q0_org);

            is_less_than_beta = (u8_q2asub_q0 < beta);

            is_less_than_beta = is_less_than_beta & is_less_than;



            {

                v8u16 is_less_than_beta_r, is_less_than_beta_l;



                is_less_than_beta_r =

                    (v8u16) __msa_sldi_b((v16i8) is_less_than_beta, zero, 8);



                q1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q1_org);

                if (!__msa_test_bz_v((v16u8) is_less_than_beta_r)) {

                    q2_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q2_org);



                    AVC_LOOP_FILTER_P1_OR_Q1(p0_org_r, q0_org_r,

                                             q1_org_r, q2_org_r,

                                             negate_tc_r, tc_r, q1_r);

                }



                is_less_than_beta_l =

                    (v8u16) __msa_sldi_b(zero, (v16i8) is_less_than_beta, 8);



                q1_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q1_org);

                if (!__msa_test_bz_v((v16u8) is_less_than_beta_l)) {

                    q2_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q2_org);



                    AVC_LOOP_FILTER_P1_OR_Q1(p0_org_l, q0_org_l,

                                             q1_org_l, q2_org_l,

                                             i16_negatetc_l, tc_l, q1_l);

                }

            }



            if (!__msa_test_bz_v(is_less_than_beta)) {

                q1 = (v16u8) __msa_pckev_b((v16i8) q1_l, (v16i8) q1_r);

                q1_org = __msa_bmnz_v(q1_org, q1, is_less_than_beta);

                STORE_UB(q1_org, data + image_width);



                is_less_than_beta = __msa_andi_b(is_less_than_beta, 1);

                tc = tc + (v16i8) is_less_than_beta;

            }



            {

                v16i8 negate_thresh, sign_negate_thresh;

                v8i16 threshold_r, threshold_l;

                v8i16 negate_thresh_l, negate_thresh_r;



                negate_thresh = zero - tc;

                sign_negate_thresh = __msa_clti_s_b(negate_thresh, 0);



                threshold_r = (v8i16) __msa_ilvr_b(zero, tc);

                negate_thresh_r = (v8i16) __msa_ilvr_b(sign_negate_thresh,

                                                       negate_thresh);



                AVC_LOOP_FILTER_P0Q0(q0_org_r, p0_org_r, p1_org_r, q1_org_r,

                                     negate_thresh_r, threshold_r, p0_r, q0_r);



                threshold_l = (v8i16) __msa_ilvl_b(zero, tc);

                negate_thresh_l = (v8i16) __msa_ilvl_b(sign_negate_thresh,

                                                       negate_thresh);



                AVC_LOOP_FILTER_P0Q0(q0_org_l, p0_org_l, p1_org_l, q1_org_l,

                                     negate_thresh_l, threshold_l, p0_l, q0_l);

            }



            p0 = (v16u8) __msa_pckev_b((v16i8) p0_l, (v16i8) p0_r);

            q0 = (v16u8) __msa_pckev_b((v16i8) q0_l, (v16i8) q0_r);



            p0_org = __msa_bmnz_v(p0_org, p0, is_less_than);

            q0_org = __msa_bmnz_v(q0_org, q0, is_less_than);



            STORE_UB(p0_org, (data - image_width));

            STORE_UB(q0_org, data);

        }

    }

}

static void avc_loopfilter_cb_or_cr_intra_edge_hor_msa(uint8_t *data_cb_or_cr,

                                                       uint8_t alpha_in,

                                                       uint8_t beta_in,

                                                       uint32_t img_width)

{

    v16u8 alpha, beta;

    v16u8 is_less_than;

    v8i16 p0_or_q0, q0_or_p0;

    v16u8 p1_or_q1_org, p0_or_q0_org, q0_or_p0_org, q1_or_p1_org;

    v16i8 zero = { 0 };

    v16u8 p0_asub_q0, p1_asub_p0, q1_asub_q0;

    v16u8 is_less_than_alpha, is_less_than_beta;

    v8i16 p1_org_r, p0_org_r, q0_org_r, q1_org_r;



    alpha = (v16u8) __msa_fill_b(alpha_in);

    beta = (v16u8) __msa_fill_b(beta_in);



    p1_or_q1_org = LOAD_UB(data_cb_or_cr - (img_width << 1));

    p0_or_q0_org = LOAD_UB(data_cb_or_cr - img_width);

    q0_or_p0_org = LOAD_UB(data_cb_or_cr);

    q1_or_p1_org = LOAD_UB(data_cb_or_cr + img_width);



    p0_asub_q0 = __msa_asub_u_b(p0_or_q0_org, q0_or_p0_org);

    p1_asub_p0 = __msa_asub_u_b(p1_or_q1_org, p0_or_q0_org);

    q1_asub_q0 = __msa_asub_u_b(q1_or_p1_org, q0_or_p0_org);



    is_less_than_alpha = (p0_asub_q0 < alpha);

    is_less_than_beta = (p1_asub_p0 < beta);

    is_less_than = is_less_than_beta & is_less_than_alpha;

    is_less_than_beta = (q1_asub_q0 < beta);

    is_less_than = is_less_than_beta & is_less_than;



    is_less_than = (v16u8) __msa_ilvr_d((v2i64) zero, (v2i64) is_less_than);



    if (!__msa_test_bz_v(is_less_than)) {

        p1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p1_or_q1_org);

        p0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p0_or_q0_org);

        q0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q0_or_p0_org);

        q1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q1_or_p1_org);



        AVC_LOOP_FILTER_P0_OR_Q0(p0_org_r, q1_org_r, p1_org_r, p0_or_q0);

        AVC_LOOP_FILTER_P0_OR_Q0(q0_org_r, p1_org_r, q1_org_r, q0_or_p0);



        p0_or_q0 = (v8i16) __msa_pckev_b(zero, (v16i8) p0_or_q0);

        q0_or_p0 = (v8i16) __msa_pckev_b(zero, (v16i8) q0_or_p0);



        p0_or_q0_org =

            __msa_bmnz_v(p0_or_q0_org, (v16u8) p0_or_q0, is_less_than);

        q0_or_p0_org =

            __msa_bmnz_v(q0_or_p0_org, (v16u8) q0_or_p0, is_less_than);



        STORE_UB(q0_or_p0_org, data_cb_or_cr);

        STORE_UB(p0_or_q0_org, data_cb_or_cr - img_width);

    }

}

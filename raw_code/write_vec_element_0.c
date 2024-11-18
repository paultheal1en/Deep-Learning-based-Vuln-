static void write_vec_element(DisasContext *s, TCGv_i64 tcg_src, int destidx,

                              int element, TCGMemOp memop)

{

    int vect_off = vec_reg_offset(destidx, element, memop & MO_SIZE);

    switch (memop) {

    case MO_8:

        tcg_gen_st8_i64(tcg_src, cpu_env, vect_off);

        break;

    case MO_16:

        tcg_gen_st16_i64(tcg_src, cpu_env, vect_off);

        break;

    case MO_32:

        tcg_gen_st32_i64(tcg_src, cpu_env, vect_off);

        break;

    case MO_64:

        tcg_gen_st_i64(tcg_src, cpu_env, vect_off);

        break;

    default:

        g_assert_not_reached();

    }

}

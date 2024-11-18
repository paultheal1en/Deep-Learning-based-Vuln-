static void gen_extu(int ot, TCGv reg)

{

    switch(ot) {

    case OT_BYTE:

        tcg_gen_ext8u_tl(reg, reg);

        break;

    case OT_WORD:

        tcg_gen_ext16u_tl(reg, reg);

        break;

    case OT_LONG:

        tcg_gen_ext32u_tl(reg, reg);

        break;

    default:

        break;

    }

}

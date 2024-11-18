static void gen_exts(int ot, TCGv reg)

{

    switch(ot) {

    case OT_BYTE:

        tcg_gen_ext8s_tl(reg, reg);

        break;

    case OT_WORD:

        tcg_gen_ext16s_tl(reg, reg);

        break;

    case OT_LONG:

        tcg_gen_ext32s_tl(reg, reg);

        break;

    default:

        break;

    }

}

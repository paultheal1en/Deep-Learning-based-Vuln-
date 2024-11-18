static inline void tcg_out_ext8u(TCGContext *s, int dest, int src)

{

    /* movzbl */

    assert(src < 4 || TCG_TARGET_REG_BITS == 64);

    tcg_out_modrm(s, OPC_MOVZBL + P_REXB_RM, dest, src);

}

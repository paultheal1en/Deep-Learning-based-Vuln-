static bool insn_crosses_page(CPUARMState *env, DisasContext *s)

{

    /* Return true if the insn at dc->pc might cross a page boundary.

     * (False positives are OK, false negatives are not.)

     */

    uint16_t insn;



    if ((s->pc & 3) == 0) {

        /* At a 4-aligned address we can't be crossing a page */

        return false;

    }



    /* This must be a Thumb insn */

    insn = arm_lduw_code(env, s->pc, s->sctlr_b);



    if ((insn >> 11) >= 0x1d) {

        /* Top five bits 0b11101 / 0b11110 / 0b11111 : this is the

         * First half of a 32-bit Thumb insn. Thumb-1 cores might

         * end up actually treating this as two 16-bit insns (see the

         * code at the start of disas_thumb2_insn()) but we don't bother

         * to check for that as it is unlikely, and false positives here

         * are harmless.

         */

        return true;

    }

    /* Definitely a 16-bit insn, can't be crossing a page. */

    return false;

}

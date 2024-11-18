static const char *target_parse_constraint(TCGArgConstraint *ct,

                                           const char *ct_str, TCGType type)

{

    switch (*ct_str++) {

    case 'r':                  /* all registers */

        ct->ct |= TCG_CT_REG;

        tcg_regset_set32(ct->u.regs, 0, 0xffff);

        break;

    case 'L':                  /* qemu_ld/st constraint */

        ct->ct |= TCG_CT_REG;

        tcg_regset_set32(ct->u.regs, 0, 0xffff);

        tcg_regset_reset_reg (ct->u.regs, TCG_REG_R2);

        tcg_regset_reset_reg (ct->u.regs, TCG_REG_R3);

        tcg_regset_reset_reg (ct->u.regs, TCG_REG_R4);

        break;

    case 'a':                  /* force R2 for division */

        ct->ct |= TCG_CT_REG;

        tcg_regset_clear(ct->u.regs);

        tcg_regset_set_reg(ct->u.regs, TCG_REG_R2);

        break;

    case 'b':                  /* force R3 for division */

        ct->ct |= TCG_CT_REG;

        tcg_regset_clear(ct->u.regs);

        tcg_regset_set_reg(ct->u.regs, TCG_REG_R3);

        break;

    case 'A':

        ct->ct |= TCG_CT_CONST_S33;

        break;

    case 'I':

        ct->ct |= TCG_CT_CONST_S16;

        break;

    case 'J':

        ct->ct |= TCG_CT_CONST_S32;

        break;

    case 'O':

        ct->ct |= TCG_CT_CONST_ORI;

        break;

    case 'X':

        ct->ct |= TCG_CT_CONST_XORI;

        break;

    case 'C':

        /* ??? We have no insight here into whether the comparison is

           signed or unsigned.  The COMPARE IMMEDIATE insn uses a 32-bit

           signed immediate, and the COMPARE LOGICAL IMMEDIATE insn uses

           a 32-bit unsigned immediate.  If we were to use the (semi)

           obvious "val == (int32_t)val" we would be enabling unsigned

           comparisons vs very large numbers.  The only solution is to

           take the intersection of the ranges.  */

        /* ??? Another possible solution is to simply lie and allow all

           constants here and force the out-of-range values into a temp

           register in tgen_cmp when we have knowledge of the actual

           comparison code in use.  */

        ct->ct |= TCG_CT_CONST_U31;

        break;

    case 'Z':

        ct->ct |= TCG_CT_CONST_ZERO;

        break;

    default:

        return NULL;

    }

    return ct_str;

}

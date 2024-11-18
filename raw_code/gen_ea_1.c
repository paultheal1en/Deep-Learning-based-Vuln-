static TCGv gen_ea(CPUM68KState *env, DisasContext *s, uint16_t insn,

                   int opsize, TCGv val, TCGv *addrp, ea_what what)

{

    TCGv reg;

    TCGv result;

    uint32_t offset;



    switch ((insn >> 3) & 7) {

    case 0: /* Data register direct.  */

        reg = DREG(insn, 0);

        if (what == EA_STORE) {

            gen_partset_reg(opsize, reg, val);

            return store_dummy;

        } else {

            return gen_extend(reg, opsize, what == EA_LOADS);

        }

    case 1: /* Address register direct.  */

        reg = AREG(insn, 0);

        if (what == EA_STORE) {

            tcg_gen_mov_i32(reg, val);

            return store_dummy;

        } else {

            return gen_extend(reg, opsize, what == EA_LOADS);

        }

    case 2: /* Indirect register */

        reg = AREG(insn, 0);

        return gen_ldst(s, opsize, reg, val, what);

    case 3: /* Indirect postincrement.  */

        reg = AREG(insn, 0);

        result = gen_ldst(s, opsize, reg, val, what);

        /* ??? This is not exception safe.  The instruction may still

           fault after this point.  */

        if (what == EA_STORE || !addrp)

            tcg_gen_addi_i32(reg, reg, opsize_bytes(opsize));

        return result;

    case 4: /* Indirect predecrememnt.  */

        {

            TCGv tmp;

            if (addrp && what == EA_STORE) {

                tmp = *addrp;

            } else {

                tmp = gen_lea(env, s, insn, opsize);

                if (IS_NULL_QREG(tmp))

                    return tmp;

                if (addrp)

                    *addrp = tmp;

            }

            result = gen_ldst(s, opsize, tmp, val, what);

            /* ??? This is not exception safe.  The instruction may still

               fault after this point.  */

            if (what == EA_STORE || !addrp) {

                reg = AREG(insn, 0);

                tcg_gen_mov_i32(reg, tmp);

            }

        }

        return result;

    case 5: /* Indirect displacement.  */

    case 6: /* Indirect index + displacement.  */

        return gen_ea_once(env, s, insn, opsize, val, addrp, what);

    case 7: /* Other */

        switch (insn & 7) {

        case 0: /* Absolute short.  */

        case 1: /* Absolute long.  */

        case 2: /* pc displacement  */

        case 3: /* pc index+displacement.  */

            return gen_ea_once(env, s, insn, opsize, val, addrp, what);

        case 4: /* Immediate.  */

            /* Sign extend values for consistency.  */

            switch (opsize) {

            case OS_BYTE:

                if (what == EA_LOADS) {

                    offset = cpu_ldsb_code(env, s->pc + 1);

                } else {

                    offset = cpu_ldub_code(env, s->pc + 1);

                }

                s->pc += 2;

                break;

            case OS_WORD:

                if (what == EA_LOADS) {

                    offset = cpu_ldsw_code(env, s->pc);

                } else {

                    offset = cpu_lduw_code(env, s->pc);

                }

                s->pc += 2;

                break;

            case OS_LONG:

                offset = read_im32(env, s);

                break;

            default:

                qemu_assert(0, "Bad immediate operand");

            }

            return tcg_const_i32(offset);

        default:

            return NULL_QREG;

        }

    }

    /* Should never happen.  */

    return NULL_QREG;

}

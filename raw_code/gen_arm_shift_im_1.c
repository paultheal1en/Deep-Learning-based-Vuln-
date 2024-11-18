static inline void gen_arm_shift_im(TCGv var, int shiftop, int shift, int flags)

{

    switch (shiftop) {

    case 0: /* LSL */

        if (shift != 0) {

            if (flags)

                shifter_out_im(var, 32 - shift);

            tcg_gen_shli_i32(var, var, shift);

        }

        break;

    case 1: /* LSR */

        if (shift == 0) {

            if (flags) {

                tcg_gen_shri_i32(var, var, 31);

                gen_set_CF(var);

            }

            tcg_gen_movi_i32(var, 0);

        } else {

            if (flags)

                shifter_out_im(var, shift - 1);

            tcg_gen_shri_i32(var, var, shift);

        }

        break;

    case 2: /* ASR */

        if (shift == 0)

            shift = 32;

        if (flags)

            shifter_out_im(var, shift - 1);

        if (shift == 32)

          shift = 31;

        tcg_gen_sari_i32(var, var, shift);

        break;

    case 3: /* ROR/RRX */

        if (shift != 0) {

            if (flags)

                shifter_out_im(var, shift - 1);

            tcg_gen_rotri_i32(var, var, shift); break;

        } else {

            TCGv tmp = load_cpu_field(CF);

            if (flags)

                shifter_out_im(var, 0);

            tcg_gen_shri_i32(var, var, 1);

            tcg_gen_shli_i32(tmp, tmp, 31);

            tcg_gen_or_i32(var, var, tmp);

            dead_tmp(tmp);

        }

    }

};

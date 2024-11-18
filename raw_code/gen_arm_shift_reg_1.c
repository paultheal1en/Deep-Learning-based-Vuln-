static inline void gen_arm_shift_reg(TCGv var, int shiftop,

                                     TCGv shift, int flags)

{

    if (flags) {

        switch (shiftop) {

        case 0: gen_helper_shl_cc(var, var, shift); break;

        case 1: gen_helper_shr_cc(var, var, shift); break;

        case 2: gen_helper_sar_cc(var, var, shift); break;

        case 3: gen_helper_ror_cc(var, var, shift); break;

        }

    } else {

        switch (shiftop) {

        case 0: gen_helper_shl(var, var, shift); break;

        case 1: gen_helper_shr(var, var, shift); break;

        case 2: gen_helper_sar(var, var, shift); break;

        case 3: tcg_gen_andi_i32(shift, shift, 0x1f);

                tcg_gen_rotr_i32(var, var, shift); break;

        }

    }

    dead_tmp(shift);

}
